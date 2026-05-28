package main

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/user/jsonrpc-server/agent"
	"github.com/user/jsonrpc-server/db"
	"github.com/user/jsonrpc-server/jsonrpc"
)

func main() {
	apiKey := os.Getenv("XIAOMI_MIMO_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "XIAOMI_MIMO_API_KEY is required")
		os.Exit(1)
	}

	// Open database in current directory.
	dbPath := filepath.Join(".", "fantasy.db")
	database, err := db.Open(dbPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to open database: %v\n", err)
		os.Exit(1)
	}
	defer database.Close()

	ag, err := agent.New(agent.Config{
		APIKey:       apiKey,
		BaseURL:      "https://token-plan-sgp.xiaomimimo.com/v1",
		DefaultModel: "xiaomi/mimo-v2.5",
		SystemPrompt: "You are a helpful coding assistant. Be concise and direct.",
		Database:     database,
		MaxHistory:   100,
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create agent: %v\n", err)
		os.Exit(1)
	}

	srv := jsonrpc.New(os.Stdin, os.Stdout)

	// ── chat.send ────────────────────────────────────────────
	srv.Register("chat.send", func(s *jsonrpc.Server, params json.RawMessage) (interface{}, error) {
		var req agent.Request
		if err := json.Unmarshal(params, &req); err != nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InvalidParams,
				Message: "Invalid params",
				Data:    err.Error(),
			}
		}
		if req.Prompt == "" {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InvalidParams,
				Message: "prompt is required",
			}
		}

		result, err := ag.Run(context.Background(), s, req)
		if err != nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InternalError,
				Message: err.Error(),
			}
		}
		return result, nil
	})

	// ── session.create ───────────────────────────────────────
	srv.Register("session.create", func(s *jsonrpc.Server, params json.RawMessage) (interface{}, error) {
		var req struct {
			ID           string `json:"id,omitempty"`
			Model        string `json:"model,omitempty"`
			SystemPrompt string `json:"systemPrompt,omitempty"`
		}
		if err := json.Unmarshal(params, &req); err != nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InvalidParams,
				Message: "Invalid params",
				Data:    err.Error(),
			}
		}

		if req.ID == "" {
			req.ID = generateID()
		}
		if req.Model == "" {
			req.Model = ag.GetDefaultModel()
		}

		session, err := ag.GetDatabase().CreateSession(req.ID, req.Model, req.SystemPrompt)
		if err != nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InternalError,
				Message: err.Error(),
			}
		}
		return session, nil
	})

	// ── session.list ─────────────────────────────────────────
	srv.Register("session.list", func(s *jsonrpc.Server, params json.RawMessage) (interface{}, error) {
		var req struct {
			Limit int `json:"limit,omitempty"`
		}
		json.Unmarshal(params, &req)

		sessions, err := ag.GetDatabase().ListSessions(req.Limit)
		if err != nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InternalError,
				Message: err.Error(),
			}
		}
		return sessions, nil
	})

	// ── session.get ──────────────────────────────────────────
	srv.Register("session.get", func(s *jsonrpc.Server, params json.RawMessage) (interface{}, error) {
		var req struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(params, &req); err != nil || req.ID == "" {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InvalidParams,
				Message: "id is required",
			}
		}

		session, err := ag.GetDatabase().GetSession(req.ID)
		if err != nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InternalError,
				Message: err.Error(),
			}
		}
		if session == nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    -32001,
				Message: "Session not found",
			}
		}

		messages, err := ag.GetDatabase().GetMessages(req.ID)
		if err != nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InternalError,
				Message: err.Error(),
			}
		}

		return map[string]any{
			"session":  session,
			"messages": messages,
		}, nil
	})

	// ── session.delete ───────────────────────────────────────
	srv.Register("session.delete", func(s *jsonrpc.Server, params json.RawMessage) (interface{}, error) {
		var req struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(params, &req); err != nil || req.ID == "" {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InvalidParams,
				Message: "id is required",
			}
		}

		if err := ag.GetDatabase().DeleteSession(req.ID); err != nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InternalError,
				Message: err.Error(),
			}
		}
		return map[string]string{"status": "deleted"}, nil
	})

	// ── set.model ────────────────────────────────────────────
	srv.Register("set.model", func(s *jsonrpc.Server, params json.RawMessage) (interface{}, error) {
		var req struct {
			Model string `json:"model"`
		}
		if err := json.Unmarshal(params, &req); err != nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InvalidParams,
				Message: "Invalid params",
				Data:    err.Error(),
			}
		}
		ag.SetDefaultModel(req.Model)
		return map[string]string{"model": req.Model}, nil
	})

	// ── set.systemPrompt ─────────────────────────────────────
	srv.Register("set.systemPrompt", func(s *jsonrpc.Server, params json.RawMessage) (interface{}, error) {
		var req struct {
			Prompt string `json:"prompt"`
		}
		if err := json.Unmarshal(params, &req); err != nil {
			return nil, &jsonrpc.ErrorObject{
				Code:    jsonrpc.InvalidParams,
				Message: "Invalid params",
				Data:    err.Error(),
			}
		}
		ag.SetSystemPrompt(req.Prompt)
		return map[string]string{"status": "ok"}, nil
	})

	// ── ping ─────────────────────────────────────────────────
	srv.Register("ping", func(s *jsonrpc.Server, params json.RawMessage) (interface{}, error) {
		return "pong", nil
	})

	fmt.Fprintf(os.Stderr, "jsonrpc agent server ready (model: %s, db: %s)\n", ag.GetDefaultModel(), dbPath)
	if err := srv.Listen(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func generateID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
}
