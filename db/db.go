package db

import (
	"database/sql"
	"fmt"
	"sync"
	"time"

	_ "modernc.org/sqlite"
)

type DB struct {
	conn *sql.DB
	mu   sync.Mutex // serializes writes
}

// Open opens (or creates) a SQLite database at the given path.
func Open(path string) (*DB, error) {
	conn, err := sql.Open("sqlite", path+"?_journal_mode=WAL&_busy_timeout=5000&_foreign_keys=ON")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}
	db := &DB{conn: conn}
	if err := db.migrate(); err != nil {
		conn.Close()
		return nil, err
	}
	return db, nil
}

func (db *DB) Close() error {
	return db.conn.Close()
}

func (db *DB) migrate() error {
	_, err := db.conn.Exec(`
		CREATE TABLE IF NOT EXISTS sessions (
			id            TEXT PRIMARY KEY,
			title         TEXT NOT NULL DEFAULT '',
			model         TEXT NOT NULL DEFAULT '',
			system_prompt TEXT NOT NULL DEFAULT '',
			created_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at    DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
		);

		CREATE TABLE IF NOT EXISTS messages (
			id           INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			role         TEXT NOT NULL CHECK(role IN ('user','assistant','tool','system')),
			content      TEXT NOT NULL DEFAULT '',
			model        TEXT NOT NULL DEFAULT '',
			tool_name    TEXT NOT NULL DEFAULT '',
			tool_call_id TEXT NOT NULL DEFAULT '',
			created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
		);

		CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
	`)
	if err != nil {
		return err
	}

	// Add model column to existing databases if missing.
	var hasModel bool
	rows, err := db.conn.Query(`PRAGMA table_info(messages)`)
	if err == nil {
		defer rows.Close()
		for rows.Next() {
			var cid int
			var name, ctype string
			var notnull int
			var dflt sql.NullString
			var pk int
			if err := rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk); err == nil && name == "model" {
				hasModel = true
				break
			}
		}
	}
	if !hasModel {
		if _, err := db.conn.Exec(`ALTER TABLE messages ADD COLUMN model TEXT NOT NULL DEFAULT ''`); err != nil {
			return fmt.Errorf("migrate model column: %w", err)
		}
	}
	return nil
}

// ── Session ────────────────────────────────────────────────

type Session struct {
	ID           string    `json:"id"`
	Title        string    `json:"title"`
	Model        string    `json:"model"`
	SystemPrompt string    `json:"systemPrompt"`
	CreatedAt    time.Time `json:"createdAt"`
	UpdatedAt    time.Time `json:"updatedAt"`
}

func (db *DB) CreateSession(id, model, systemPrompt string) (*Session, error) {
	db.mu.Lock()
	defer db.mu.Unlock()
	now := time.Now()
	_, err := db.conn.Exec(
		`INSERT INTO sessions (id, model, system_prompt, created_at, updated_at) VALUES (?, ?, ?, ?, ?)`,
		id, model, systemPrompt, now, now,
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	return &Session{ID: id, Model: model, SystemPrompt: systemPrompt, CreatedAt: now, UpdatedAt: now}, nil
}

func (db *DB) GetSession(id string) (*Session, error) {
	var s Session
	err := db.conn.QueryRow(
		`SELECT id, title, model, system_prompt, created_at, updated_at FROM sessions WHERE id = ?`, id,
	).Scan(&s.ID, &s.Title, &s.Model, &s.SystemPrompt, &s.CreatedAt, &s.UpdatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("get session: %w", err)
	}
	return &s, nil
}

func (db *DB) ListSessions(limit int) ([]Session, error) {
	if limit <= 0 {
		limit = 50
	}
	rows, err := db.conn.Query(
		`SELECT id, title, model, system_prompt, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?`, limit,
	)
	if err != nil {
		return nil, fmt.Errorf("list sessions: %w", err)
	}
	defer rows.Close()

	var sessions []Session
	for rows.Next() {
		var s Session
		if err := rows.Scan(&s.ID, &s.Title, &s.Model, &s.SystemPrompt, &s.CreatedAt, &s.UpdatedAt); err != nil {
			return nil, err
		}
		sessions = append(sessions, s)
	}
	return sessions, rows.Err()
}

func (db *DB) UpdateSessionTitle(id, title string) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	return db.updateSessionTitle(id, title)
}

func (db *DB) updateSessionTitle(id, title string) error {
	_, err := db.conn.Exec(`UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?`, title, time.Now(), id)
	return err
}

func (db *DB) TouchSession(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	return db.touchSession(id)
}

func (db *DB) touchSession(id string) error {
	_, err := db.conn.Exec(`UPDATE sessions SET updated_at = ? WHERE id = ?`, time.Now(), id)
	return err
}

func (db *DB) DeleteSession(id string) error {
	db.mu.Lock()
	defer db.mu.Unlock()
	_, err := db.conn.Exec(`DELETE FROM sessions WHERE id = ?`, id)
	return err
}

// ── Message ────────────────────────────────────────────────

type Message struct {
	ID         int64     `json:"id"`
	SessionID  string    `json:"sessionId"`
	Role       string    `json:"role"`
	Content    string    `json:"content"`
	Model      string    `json:"model,omitempty"`
	ToolName   string    `json:"toolName,omitempty"`
	ToolCallID string    `json:"toolCallId,omitempty"`
	CreatedAt  time.Time `json:"createdAt"`
}

func (db *DB) AddMessage(sessionID, role, content string) (*Message, error) {
	return db.AddMessageWithTool(sessionID, role, content, "", "", "")
}

func (db *DB) AddMessageWithTool(sessionID, role, content, model, toolName, toolCallID string) (*Message, error) {
	db.mu.Lock()
	defer db.mu.Unlock()
	now := time.Now()
	res, err := db.conn.Exec(
		`INSERT INTO messages (session_id, role, content, model, tool_name, tool_call_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)`,
		sessionID, role, content, model, toolName, toolCallID, now,
	)
	if err != nil {
		return nil, fmt.Errorf("add message: %w", err)
	}
	id, _ := res.LastInsertId()
	_ = db.touchSession(sessionID)
	return &Message{ID: id, SessionID: sessionID, Role: role, Content: content, Model: model, ToolName: toolName, ToolCallID: toolCallID, CreatedAt: now}, nil
}

func (db *DB) GetMessages(sessionID string) ([]Message, error) {
	rows, err := db.conn.Query(
		`SELECT id, session_id, role, content, model, tool_name, tool_call_id, created_at FROM messages WHERE session_id = ? ORDER BY id ASC`, sessionID,
	)
	if err != nil {
		return nil, fmt.Errorf("get messages: %w", err)
	}
	defer rows.Close()

	var msgs []Message
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ID, &m.SessionID, &m.Role, &m.Content, &m.Model, &m.ToolName, &m.ToolCallID, &m.CreatedAt); err != nil {
			return nil, err
		}
		msgs = append(msgs, m)
	}
	return msgs, rows.Err()
}

// GetMessagesPaginated returns messages with limit/offset pagination.
// offset=0, limit=0 returns all messages.
func (db *DB) GetMessagesPaginated(sessionID string, offset, limit int) ([]Message, int, error) {
	// Get total count.
	total, err := db.GetMessageCount(sessionID)
	if err != nil {
		return nil, 0, err
	}

	if limit <= 0 {
		limit = total
	}
	if offset < 0 {
		offset = 0
	}

	rows, err := db.conn.Query(
		`SELECT id, session_id, role, content, model, tool_name, tool_call_id, created_at FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT ? OFFSET ?`,
		sessionID, limit, offset,
	)
	if err != nil {
		return nil, 0, fmt.Errorf("get messages paginated: %w", err)
	}
	defer rows.Close()

	var msgs []Message
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ID, &m.SessionID, &m.Role, &m.Content, &m.Model, &m.ToolName, &m.ToolCallID, &m.CreatedAt); err != nil {
			return nil, 0, err
		}
		msgs = append(msgs, m)
	}
	return msgs, total, rows.Err()
}

func (db *DB) GetMessageCount(sessionID string) (int, error) {
	var count int
	err := db.conn.QueryRow(`SELECT COUNT(*) FROM messages WHERE session_id = ?`, sessionID).Scan(&count)
	return count, err
}

// DeleteMessagesBefore deletes all messages with id < beforeID for a session.
func (db *DB) DeleteMessagesBefore(sessionID string, beforeID int64) (int64, error) {
	db.mu.Lock()
	defer db.mu.Unlock()
	res, err := db.conn.Exec(`DELETE FROM messages WHERE session_id = ? AND id < ?`, sessionID, beforeID)
	if err != nil {
		return 0, fmt.Errorf("delete messages: %w", err)
	}
	n, _ := res.RowsAffected()
	_ = db.touchSession(sessionID)
	return n, nil
}
