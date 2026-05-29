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
	Prompt    string     `json:"prompt"`
	SessionID string     `json:"sessionId,omitempty"`
	Model     string     `json:"model,omitempty"`
	Files     []FilePart `json:"files,omitempty"`
}

// FilePart is a base64-encoded file attachment.
type FilePart struct {
	Filename  string `json:"filename"`
	Data      []byte `json:"data"`
	MediaType string `json:"mediaType"`
}

const maxFileSize = 5 * 1024 * 1024 // 5MB

// ModelInfo holds context limits for a model.
type ModelInfo struct {
	ContextWindow int64 `json:"contextWindow"` // max input tokens
	MaxOutput     int64 `json:"maxOutput"`      // max output tokens
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
	mu             sync.Mutex
	provider       fantasy.Provider
	database       *db.DB
	defaultModel   string
	smallModel     string
	multimodalModel string
	modelInfo      map[string]ModelInfo // keyed by model ID
	apiKey         string
	baseURL        string
	systemPrompt   string
	maxHistory     int // fallback message count limit (0 = no limit)
}

// Config holds the initial configuration for the agent.
type Config struct {
	APIKey          string
	BaseURL         string
	DefaultModel    string
	SmallModel      string
	MultimodalModel string
	ModelInfo       map[string]ModelInfo // keyed by model ID
	SystemPrompt    string
	Database        *db.DB
	MaxHistory      int // fallback message count limit (0 = no limit)
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
		provider:        provider,
		database:        cfg.Database,
		defaultModel:    cfg.DefaultModel,
		smallModel:      cfg.SmallModel,
		multimodalModel: cfg.MultimodalModel,
		modelInfo:       cfg.ModelInfo,
		apiKey:          cfg.APIKey,
		baseURL:         cfg.BaseURL,
		systemPrompt:    cfg.SystemPrompt,
		maxHistory:      cfg.MaxHistory,
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

// SetSmallModel changes the small/fast model.
func (a *Agent) SetSmallModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.smallModel = model
}

// GetSmallModel returns the current small model.
func (a *Agent) GetSmallModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.smallModel
}

// SetMultimodalModel changes the multimodal model.
func (a *Agent) SetMultimodalModel(model string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.multimodalModel = model
}

// GetMultimodalModel returns the current multimodal model.
func (a *Agent) GetMultimodalModel() string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.multimodalModel
}

// GetModels returns the current model configuration.
func (a *Agent) GetModels() map[string]string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return map[string]string{
		"primary":    a.defaultModel,
		"small":      a.smallModel,
		"multimodal": a.multimodalModel,
	}
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

// estimateTokens returns a rough token count estimate for a string.
// Uses ~4 chars per token as a heuristic.
func estimateTokens(s string) int64 {
	return int64(len(s)+3) / 4
}

// messageTokens estimates the token count of a single message.
func messageTokens(m fantasy.Message) int64 {
	var total int64
	for _, p := range m.Content {
		if t, ok := p.(fantasy.TextPart); ok {
			total += estimateTokens(t.Text)
		}
	}
	return total + 4 // overhead for role/metadata
}

// historyTokens estimates the total token count of a message history.
func historyTokens(history []fantasy.Message) int64 {
	var total int64
	for _, m := range history {
		total += messageTokens(m)
	}
	return total
}

// contextBudget returns the token budget available for conversation history
// given a model's context window and max output. Uses 80% of (context - output)
// to leave room for system prompt and tools.
func (a *Agent) contextBudget(modelID string) int64 {
	info, ok := a.modelInfo[modelID]
	if !ok || info.ContextWindow == 0 {
		return 0 // no limit configured
	}
	return (info.ContextWindow - info.MaxOutput) * 80 / 100
}

// CompactHistory is the public entry point for manual compaction.
func (a *Agent) CompactHistory(ctx context.Context, sessionID string) error {
	history, err := a.buildHistory(sessionID)
	if err != nil {
		return err
	}

	a.mu.Lock()
	modelID := a.defaultModel
	a.mu.Unlock()

	compacted, err := a.compactHistory(ctx, modelID, history)
	if err != nil {
		return err
	}

	// If history changed, we compacted. The summary is already in the history
	// as a system message — no need to persist it, it'll be used on next request.
	_ = compacted
	return nil
}

// compactHistory summarizes old messages when history exceeds the model's
// token budget. Uses the small model for summarization.
func (a *Agent) compactHistory(ctx context.Context, modelID string, history []fantasy.Message) ([]fantasy.Message, error) {
	budget := a.contextBudget(modelID)

	// Fallback to message count if no model info configured.
	if budget == 0 {
		if a.maxHistory > 0 && len(history) > a.maxHistory {
			history = history[len(history)-a.maxHistory:]
			history = append([]fantasy.Message{{
				Role:    fantasy.MessageRoleSystem,
				Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Earlier conversation context has been truncated. The following messages are the most recent portion of an ongoing conversation."}},
			}}, history...)
		}
		return history, nil
	}

	// Check if we're within budget.
	totalTokens := historyTokens(history)
	if totalTokens <= budget {
		return history, nil
	}

	a.mu.Lock()
	smallID := a.smallModel
	a.mu.Unlock()

	if smallID == "" {
		// No small model, truncate by estimated ratio.
		ratio := float64(budget) / float64(totalTokens)
		cutoff := int(float64(len(history)) * ratio)
		if cutoff < 1 {
			cutoff = 1
		}
		history = history[len(history)-cutoff:]
		history = append([]fantasy.Message{{
			Role:    fantasy.MessageRoleSystem,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Earlier conversation context has been truncated. The following messages are the most recent portion of an ongoing conversation."}},
		}}, history...)
		return history, nil
	}

	// Progressive summarization: summarize from the oldest messages until
	// we fit within the budget.
	smallModel, err := a.provider.LanguageModel(ctx, smallID)
	if err != nil {
		return nil, fmt.Errorf("small model %q: %w", smallID, err)
	}

	summarizer := fantasy.NewAgent(smallModel,
		fantasy.WithSystemPrompt("You are a conversation summarizer. Summarize the following conversation history concisely, preserving key facts, decisions, code changes, and context needed to continue the conversation. Output only the summary, no preamble."),
	)

	for totalTokens > budget && len(history) > 2 {
		// Find cutoff: remove ~25% of messages from the start each round.
		cutoff := len(history) / 4
		if cutoff < 1 {
			cutoff = 1
		}

		oldMessages := history[:cutoff]
		remaining := history[cutoff:]

		// Format old messages for summarization.
		var buf strings.Builder
		for _, m := range oldMessages {
			switch m.Role {
			case fantasy.MessageRoleUser:
				buf.WriteString("User: ")
			case fantasy.MessageRoleAssistant:
				buf.WriteString("Assistant: ")
			case fantasy.MessageRoleTool:
				buf.WriteString("Tool: ")
			case fantasy.MessageRoleSystem:
				buf.WriteString("System: ")
			}
			for _, part := range m.Content {
				if t, ok := part.(fantasy.TextPart); ok {
					buf.WriteString(t.Text)
				}
			}
			buf.WriteString("\n")
		}

		result, err := summarizer.Generate(ctx, fantasy.AgentCall{
			Prompt: buf.String(),
		})
		if err != nil {
			return nil, fmt.Errorf("summarize history: %w", err)
		}

		summary := strings.TrimSpace(result.Response.Content.Text())

		// Replace old messages with summary.
		history = make([]fantasy.Message, 0, 1+len(remaining))
		history = append(history, fantasy.Message{
			Role:    fantasy.MessageRoleSystem,
			Content: []fantasy.MessagePart{fantasy.TextPart{Text: "Conversation summary (earlier messages were compacted):\n" + summary}},
		})
		history = append(history, remaining...)

		totalTokens = historyTokens(history)
	}

	return history, nil
}

// Run executes an agent request, streaming notifications via the JSON-RPC server.
func (a *Agent) Run(ctx context.Context, srv *jsonrpc.Server, req Request) (*Response, error) {
	a.mu.Lock()
	modelID := a.defaultModel
	multimodalID := a.multimodalModel
	systemPrompt := a.systemPrompt
	a.mu.Unlock()

	if req.Model != "" {
		modelID = req.Model
	} else if len(req.Files) > 0 && multimodalID != "" {
		// Auto-select multimodal model when files are attached.
		modelID = multimodalID
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

	// Compact history if it exceeds limit (summarizes old messages via small model).
	history, err = a.compactHistory(ctx, modelID, history)
	if err != nil {
		return nil, fmt.Errorf("compact history: %w", err)
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

	// Convert request files to fantasy FileParts, validating size.
	var files []fantasy.FilePart
	for _, f := range req.Files {
		if len(f.Data) > maxFileSize {
			return nil, fmt.Errorf("file %q exceeds 5MB limit (%d bytes)", f.Filename, len(f.Data))
		}
		files = append(files, fantasy.FilePart{
			Filename:  f.Filename,
			Data:      f.Data,
			MediaType: f.MediaType,
		})
	}

	// When files are present, the fantasy framework requires Prompt to be set.
	// We also remove the last user message from history to avoid duplication,
	// since the framework will create a user message from Prompt + Files.
	streamPrompt := ""
	if len(files) > 0 {
		streamPrompt = req.Prompt
		if len(history) > 0 && history[len(history)-1].Role == fantasy.MessageRoleUser {
			history = history[:len(history)-1]
		}
	}

	streamCall := fantasy.AgentStreamCall{
		Prompt:   streamPrompt,
		Messages: history,
		Files:    files,

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
