package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/user/jsonrpc-server/agent"
	"github.com/user/jsonrpc-server/jsonrpc"
)

func main() {
	apiKey := os.Getenv("XIAOMI_MIMO_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "XIAOMI_MIMO_API_KEY is required")
		os.Exit(1)
	}

	ag, err := agent.New(agent.Config{
		APIKey:       apiKey,
		BaseURL:      "https://token-plan-sgp.xiaomimimo.com/v1",
		DefaultModel: "xiaomi/mimo-v2.5",
		SystemPrompt: "You are a helpful coding assistant. Be concise and direct.",
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to create agent: %v\n", err)
		os.Exit(1)
	}

	srv := jsonrpc.New(os.Stdin, os.Stdout)

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

	srv.Register("ping", func(s *jsonrpc.Server, params json.RawMessage) (interface{}, error) {
		return "pong", nil
	})

	fmt.Fprintf(os.Stderr, "jsonrpc agent server ready (model: %s)\n", ag.GetDefaultModel())
	if err := srv.Listen(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}
