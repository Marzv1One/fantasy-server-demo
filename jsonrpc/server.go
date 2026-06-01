package jsonrpc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
)

type Request struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
	ID      *int            `json:"id,omitempty"`
}

type Response struct {
	JSONRPC string        `json:"jsonrpc"`
	Result  interface{}   `json:"result,omitempty"`
	Error   *ErrorObject  `json:"error,omitempty"`
	ID      *int          `json:"id"`
}

type Notification struct {
	JSONRPC string      `json:"jsonrpc"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

type ErrorObject struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

func (e *ErrorObject) Error() string {
	return e.Message
}

// Standard JSON-RPC 2.0 error codes.
const (
	ParseError     = -32700
	InvalidRequest = -32600
	MethodNotFound = -32601
	InvalidParams  = -32602
	InternalError  = -32603
)

// HandlerFunc receives the server (for sending notifications) and the request params.
type HandlerFunc func(s *Server, params json.RawMessage) (interface{}, error)

type Server struct {
	r              io.Reader
	w              io.Writer
	enc            *json.Encoder
	mu             sync.Mutex
	wg             sync.WaitGroup
	methods        map[string]HandlerFunc
	pendingReqMu   sync.Mutex
	pendingReq     map[int]chan json.RawMessage
	reqCounter     atomic.Int32
}

func New(r io.Reader, w io.Writer) *Server {
	return &Server{
		r:          r,
		w:          w,
		enc:        json.NewEncoder(w),
		methods:    make(map[string]HandlerFunc),
		pendingReq: make(map[int]chan json.RawMessage),
	}
}

// Register adds a method handler that will be called when a matching request arrives.
func (s *Server) Register(method string, fn HandlerFunc) {
	s.methods[method] = fn
}

// Notify sends a JSON-RPC notification (no id) to the client. Thread-safe.
func (s *Server) Notify(method string, params interface{}) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.enc.Encode(Notification{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	})
}

// Request sends a JSON-RPC request to the client and waits for the response. Thread-safe.
func (s *Server) Request(method string, params interface{}) (json.RawMessage, error) {
	id := int(s.reqCounter.Add(1))
	ch := make(chan json.RawMessage, 1)

	s.pendingReqMu.Lock()
	s.pendingReq[id] = ch
	s.pendingReqMu.Unlock()

	defer func() {
		s.pendingReqMu.Lock()
		delete(s.pendingReq, id)
		s.pendingReqMu.Unlock()
	}()

	s.mu.Lock()
	err := s.enc.Encode(Request{
		JSONRPC: "2.0",
		Method:  method,
		Params:  mustMarshal(params),
		ID:      &id,
	})
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}

	resp := <-ch
	// Check if the response is an error.
	var envelope struct {
		Result json.RawMessage `json:"result"`
		Error  *ErrorObject    `json:"error"`
	}
	if err := json.Unmarshal(resp, &envelope); err != nil {
		return nil, fmt.Errorf("invalid response: %w", err)
	}
	if envelope.Error != nil {
		return nil, envelope.Error
	}
	return envelope.Result, nil
}

// RequestWithContext sends a JSON-RPC request and waits for the response or context cancellation.
func (s *Server) RequestWithContext(ctx context.Context, method string, params interface{}) (json.RawMessage, error) {
	id := int(s.reqCounter.Add(1))
	ch := make(chan json.RawMessage, 1)

	s.pendingReqMu.Lock()
	s.pendingReq[id] = ch
	s.pendingReqMu.Unlock()

	defer func() {
		s.pendingReqMu.Lock()
		delete(s.pendingReq, id)
		s.pendingReqMu.Unlock()
	}()

	s.mu.Lock()
	err := s.enc.Encode(Request{
		JSONRPC: "2.0",
		Method:  method,
		Params:  mustMarshal(params),
		ID:      &id,
	})
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case resp := <-ch:
		var envelope struct {
			Result json.RawMessage `json:"result"`
			Error  *ErrorObject    `json:"error"`
		}
		if err := json.Unmarshal(resp, &envelope); err != nil {
			return nil, fmt.Errorf("invalid response: %w", err)
		}
		if envelope.Error != nil {
			return nil, envelope.Error
		}
		return envelope.Result, nil
	}
}

func mustMarshal(v interface{}) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

// Listen reads requests from stdin and writes responses to stdout. Blocks until EOF.
func (s *Server) Listen() error {
	dec := json.NewDecoder(s.r)

	for {
		var raw json.RawMessage
		if err := dec.Decode(&raw); err != nil {
			if err == io.EOF {
				s.wg.Wait()
				return nil
			}
			s.writeError(nil, ParseError, "Parse error", nil)
			continue
		}

		// Detect if this is a response (has "id" + "result"/"error", no "method")
		var probe struct {
			ID     *int            `json:"id"`
			Method string          `json:"method"`
			Result json.RawMessage `json:"result"`
			Error  *ErrorObject    `json:"error"`
		}
		if err := json.Unmarshal(raw, &probe); err == nil && probe.ID != nil && probe.Method == "" {
			// This is a response to a server-initiated request.
			s.pendingReqMu.Lock()
			ch, ok := s.pendingReq[*probe.ID]
			s.pendingReqMu.Unlock()
			if ok {
				ch <- raw
			}
			continue
		}

		// Otherwise it's a client request.
		var req Request
		if err := json.Unmarshal(raw, &req); err != nil {
			s.writeError(nil, ParseError, "Parse error", nil)
			continue
		}

		s.wg.Add(1)
		go s.handle(req)
	}
}

func (s *Server) handle(req Request) {
	defer s.wg.Done()

	if req.JSONRPC != "2.0" {
		s.writeError(req.ID, InvalidRequest, "Invalid JSON-RPC version", nil)
		return
	}

	fn, ok := s.methods[req.Method]
	if !ok {
		s.writeError(req.ID, MethodNotFound, fmt.Sprintf("Method not found: %s", req.Method), nil)
		return
	}

	result, err := fn(s, req.Params)
	if err != nil {
		rpcErr, ok := err.(*ErrorObject)
		if !ok {
			rpcErr = &ErrorObject{
				Code:    InternalError,
				Message: err.Error(),
			}
		}
		s.writeError(req.ID, rpcErr.Code, rpcErr.Message, rpcErr.Data)
		return
	}

	if req.ID == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.enc.Encode(Response{
		JSONRPC: "2.0",
		Result:  result,
		ID:      req.ID,
	})
}

func (s *Server) writeError(id *int, code int, msg string, data interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.enc.Encode(Response{
		JSONRPC: "2.0",
		Error: &ErrorObject{
			Code:    code,
			Message: msg,
			Data:    data,
		},
		ID: id,
	})
}
