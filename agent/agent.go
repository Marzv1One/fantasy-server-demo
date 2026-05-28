package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"sync"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"

	"github.com/user/jsonrpc-server/db"
	"github.com/user/jsonrpc-server/jsonrpc"
	"github.com/user/jsonrpc-server/tools"
)

// Request is the payload for the chat.send JSON-RPC method.
type Request struct {
	Prompt    string `json:"prompt"`
	SessionID string `json:"sessionId,omitempty"`
	Model     string `json:"model,omitempty"`
}

// Response is returned after the agent finishes.
type Response struct {
	Text      string `json:"text"`
	SessionID string `json:"sessionId"`
	Usage     Usage  `json:"usage"`
}

type Usage struct {
	InputTokens  int64 `json:"inputTokens"`
	OutputTokens int64 `json:"outputTokens"`
	TotalTokens  int64 `json:"totalTokens"`
}

// Agent wraps a fantasy agent and manages conversations.
type Agent struct {
	mu           sync.Mutex
	provider     fantasy.Provider
	database     *db.DB
	defaultModel string
	apiKey       string
	baseURL      string
	systemPrompt string
}

// Config holds the initial configuration for the agent.
type Config struct {
	APIKey       string
	BaseURL      string
	DefaultModel string
	SystemPrompt string
	Database     *db.DB
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
		database:     cfg.Database,
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

// GetDatabase returns the database instance.
func (a *Agent) GetDatabase() *db.DB {
	return a.database
}

func generateID() string {
	b := make([]byte, 8)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// ensureSession returns the session ID, creating a new session if needed.
func (a *Agent) ensureSession(sessionID, model string) (string, error) {
	if sessionID != "" {
		existing, err := a.database.GetSession(sessionID)
		if err != nil {
			return "", fmt.Errorf("check session: %w", err)
		}
		if existing != nil {
			return sessionID, nil
		}
	}

	// Create new session.
	if sessionID == "" {
		sessionID = generateID()
	}
	a.mu.Lock()
	sysPrompt := a.systemPrompt
	a.mu.Unlock()
	if _, err := a.database.CreateSession(sessionID, model, sysPrompt); err != nil {
		return "", fmt.Errorf("create session: %w", err)
	}
	return sessionID, nil
}

// buildHistory loads conversation history from DB and returns fantasy messages.
func (a *Agent) buildHistory(sessionID string) ([]fantasy.Message, error) {
	msgs, err := a.database.GetMessages(sessionID)
	if err != nil {
		return nil, err
	}

	var messages []fantasy.Message
	for _, m := range msgs {
		switch m.Role {
		case "user":
			messages = append(messages, fantasy.NewUserMessage(m.Content))
		case "assistant":
			messages = append(messages, fantasy.Message{
				Role:    fantasy.MessageRoleAssistant,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: m.Content}},
			})
		case "tool":
			messages = append(messages, fantasy.Message{
				Role:    fantasy.MessageRoleTool,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: m.Content}},
			})
		}
	}
	return messages, nil
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

	// Ensure session exists.
	sessionID, err := a.ensureSession(req.SessionID, modelID)
	if err != nil {
		return nil, err
	}

	// Save user message.
	if _, err := a.database.AddMessage(sessionID, "user", req.Prompt); err != nil {
		return nil, fmt.Errorf("save user message: %w", err)
	}

	// Load conversation history.
	history, err := a.buildHistory(sessionID)
	if err != nil {
		return nil, fmt.Errorf("load history: %w", err)
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
		Messages: history,

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

	// Save assistant response.
	if finalText != "" {
		if _, err := a.database.AddMessage(sessionID, "assistant", finalText); err != nil {
			return nil, fmt.Errorf("save assistant message: %w", err)
		}
	}

	// Auto-title: use first 80 chars of first message if title is empty.
	count, _ := a.database.GetMessageCount(sessionID)
	if count <= 2 {
		title := strings.ReplaceAll(req.Prompt, "\n", " ")
		title = strings.ReplaceAll(title, "\r", "")
		if len(title) > 80 {
			title = title[:80] + "..."
		}
		a.database.UpdateSessionTitle(sessionID, title)
	}

	return &Response{
		Text:      finalText,
		SessionID: sessionID,
		Usage:     totalUsage,
	}, nil
}
