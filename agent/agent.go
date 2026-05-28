package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"

	"github.com/user/jsonrpc-server/jsonrpc"
	"github.com/user/jsonrpc-server/tools"
)

// Request is the payload for the chat.send JSON-RPC method.
type Request struct {
	Prompt string `json:"prompt"`
	Model  string `json:"model,omitempty"`
}

// Response is returned after the agent finishes.
type Response struct {
	Text  string `json:"text"`
	Usage Usage  `json:"usage"`
}

type Usage struct {
	InputTokens  int64 `json:"inputTokens"`
	OutputTokens int64 `json:"outputTokens"`
	TotalTokens  int64 `json:"totalTokens"`
}

// Agent wraps a fantasy agent and manages conversations.
type Agent struct {
	mu            sync.Mutex
	provider      fantasy.Provider
	defaultModel  string
	apiKey        string
	baseURL       string
	systemPrompt  string
}

// Config holds the initial configuration for the agent.
type Config struct {
	APIKey       string
	BaseURL      string
	DefaultModel string
	SystemPrompt string
}

// New creates a new Agent.
func New(cfg Config) (*Agent, error) {
	provider, err := openaicompat.New(
		openaicompat.WithBaseURL(cfg.BaseURL),
		openaicompat.WithAPIKey(cfg.APIKey),
	)
	if err != nil {
		return nil, fmt.Errorf("create provider: %w", err)
	}

	return &Agent{
		provider:     provider,
		defaultModel: cfg.DefaultModel,
		apiKey:       cfg.APIKey,
		baseURL:      cfg.BaseURL,
		systemPrompt: cfg.SystemPrompt,
	}, nil
}

// SetDefaultModel changes the default model used when none is specified.
func (a *Agent) SetDefaultModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.defaultModel = model
}

// GetDefaultModel returns the current default model.
func (a *Agent) GetDefaultModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.defaultModel
}

// SetSystemPrompt changes the system prompt.
func (a *Agent) SetSystemPrompt(prompt string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.systemPrompt = prompt
}

// Run executes an agent request, streaming notifications via the JSON-RPC server.
func (a *Agent) Run(ctx context.Context, srv *jsonrpc.Server, req Request) (*Response, error) {
	a.mu.Lock()
	modelID := a.defaultModel
	systemPrompt := a.systemPrompt
	a.mu.Unlock()

	if req.Model != "" {
		modelID = req.Model
	}

	model, err := a.provider.LanguageModel(ctx, modelID)
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", modelID, err)
	}

	notify := func(method string, params interface{}) error {
		return srv.Notify(method, params)
	}

	agentTools := tools.RegisterAll(notify)

	ag := fantasy.NewAgent(model,
		fantasy.WithSystemPrompt(systemPrompt),
		fantasy.WithTools(agentTools...),
		fantasy.WithStopConditions(fantasy.StepCountIs(20)),
	)

	var finalText string
	var totalUsage Usage

	streamCall := fantasy.AgentStreamCall{
		Prompt: req.Prompt,

		OnTextDelta: func(id, text string) error {
			finalText += text
			return notify("chat.textDelta", map[string]any{
				"text": text,
			})
		},

		OnToolCall: func(tc fantasy.ToolCallContent) error {
			return notify("chat.toolCall", tools.ToolCallNotification{
				ToolName: tc.ToolName,
				Input:    json.RawMessage(tc.Input),
			})
		},

		OnToolResult: func(res fantasy.ToolResultContent) error {
			var output string
			var isError bool
			if text, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentText](res.Result); ok {
				output = text.Text
			} else if errRes, ok := fantasy.AsToolResultOutputType[fantasy.ToolResultOutputContentError](res.Result); ok {
				output = errRes.Error.Error()
				isError = true
			}
			return notify("chat.toolResult", tools.ToolResultNotification{
				ToolName: res.ToolName,
				Output:   output,
				IsError:  isError,
			})
		},

		OnStepFinish: func(step fantasy.StepResult) error {
			return notify("chat.stepFinish", map[string]any{
				"finishReason": step.FinishReason,
			})
		},

		OnFinish: func(result *fantasy.AgentResult) {
			totalUsage = Usage{
				InputTokens:  result.TotalUsage.InputTokens,
				OutputTokens: result.TotalUsage.OutputTokens,
				TotalTokens:  result.TotalUsage.TotalTokens,
			}
		},

		OnError: func(err error) {
			notify("chat.error", map[string]any{
				"error": err.Error(),
			})
		},
	}

	result, err := ag.Stream(ctx, streamCall)
	if err != nil {
		return nil, fmt.Errorf("agent stream: %w", err)
	}

	totalUsage = Usage{
		InputTokens:  result.TotalUsage.InputTokens,
		OutputTokens: result.TotalUsage.OutputTokens,
		TotalTokens:  result.TotalUsage.TotalTokens,
	}

	return &Response{
		Text:  finalText,
		Usage: totalUsage,
	}, nil
}
