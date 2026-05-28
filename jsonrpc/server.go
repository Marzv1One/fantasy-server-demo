package jsonrpc

import (
	"encoding/json"
	"fmt"
	"io"
	"sync"
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
	r       io.Reader
	w       io.Writer
	enc     *json.Encoder
	mu      sync.Mutex
	wg      sync.WaitGroup
	methods map[string]HandlerFunc
}

func New(r io.Reader, w io.Writer) *Server {
	return &Server{
		r:       r,
		w:       w,
		enc:     json.NewEncoder(w),
		methods: make(map[string]HandlerFunc),
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

// Listen reads requests from stdin and writes responses to stdout. Blocks until EOF.
func (s *Server) Listen() error {
	dec := json.NewDecoder(s.r)

	for {
		var req Request
		if err := dec.Decode(&req); err != nil {
			if err == io.EOF {
				s.wg.Wait()
				return nil
			}
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
