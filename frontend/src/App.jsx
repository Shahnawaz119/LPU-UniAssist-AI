import React, { useState, useEffect, useRef } from 'react'
import {
  Send,
  Bot,
  User,
  GraduationCap,
  RotateCcw,
  BookOpen,
  ChevronDown,
  ChevronUp,
  ShieldCheck,
  AlertTriangle,
  ExternalLink
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const QUICK_PROMPTS = [
  "How much attendance is mandatory at LPU to appear in ETE exams?",
  "Can I pay my tuition fee in installments via UMS?",
  "How to apply for an online night gate pass on UMS / LPU Touch?",
  "How do I connect to LPU-Student-WiFi and register my device on NetReg?"
]

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId, setSessionId] = useState(() => 'sess_' + Math.random().toString(36).substr(2, 9))
  const [backendHealth, setBackendHealth] = useState(null)
  const [expandedSources, setExpandedSources] = useState({})

  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  // Fetch backend health on mount
  useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then(res => res.json())
      .then(data => setBackendHealth(data))
      .catch(err => {
        console.warn('Backend not yet reachable:', err)
        setBackendHealth({ status: 'offline' })
      })
  }, [])

  const toggleSources = (index) => {
    setExpandedSources(prev => ({
      ...prev,
      [index]: !prev[index]
    }))
  }

  const handleResetChat = async () => {
    try {
      await fetch(`${API_BASE_URL}/api/chat/${sessionId}`, { method: 'DELETE' })
    } catch (e) {
      // ignore
    }
    const newSession = 'sess_' + Math.random().toString(36).substr(2, 9)
    setSessionId(newSession)
    setMessages([])
    setExpandedSources({})
  }

  const handleSubmit = async (e) => {
    e?.preventDefault()
    const question = input.trim()
    if (!question || isLoading) return

    setInput('')
    await handleSend(question)
  }

  const handleSend = async (questionText) => {
    const userMsg = {
      id: Date.now(),
      role: 'user',
      content: questionText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    }

    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: questionText,
          session_id: sessionId
        })
      })

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`)
      }

      const data = await response.json()

      const botMsg = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
        error: data.error || false,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }

      setMessages(prev => [...prev, botMsg])
    } catch (error) {
      console.error('Error contacting backend:', error)
      const errorMsg = {
        id: Date.now() + 1,
        role: 'assistant',
        content: `Could not connect to the University Helpdesk server (${error.message}). Please ensure the backend is running.`,
        sources: [],
        error: true,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
      setMessages(prev => [...prev, errorMsg])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <GraduationCap size={28} className="text-blue-400" />
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h1>LPU UniAssist AI</h1>
              <span className="badge">LPU Helpdesk</span>
            </div>
            <p style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
              Lovely Professional University (LPU) AI Assistant & UMS Guide
            </p>
          </div>
        </div>

        <div className="header-actions">
          {backendHealth && (
            <span
              style={{
                fontSize: '0.75rem',
                padding: '0.2rem 0.5rem',
                borderRadius: '4px',
                backgroundColor: backendHealth.status === 'healthy' ? '#065f46' : '#831843',
                color: '#ffffff'
              }}
            >
              {backendHealth.status === 'healthy'
                ? `● ${backendHealth.model?.split('/')[1] || backendHealth.model}`
                : 'Offline'}
            </span>
          )}

          <button
            onClick={handleResetChat}
            className="btn-secondary"
            title="Start new conversation"
          >
            <RotateCcw size={15} />
            <span>New Chat</span>
          </button>
        </div>
      </header>

      {/* Main Chat Area */}
      <main className="chat-main">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">
              <BookOpen size={28} />
            </div>
            <h2>Welcome to LPU UniAssist AI</h2>
            <p>
              Ask any question about Lovely Professional University (LPU) policies, UMS, LPUNEST,
              tuition installments, 75% attendance rule, hostel gate passes, or campus Wi-Fi. Every answer
              is strictly grounded in official LPU records.
            </p>

            <div className="quick-prompts">
              {QUICK_PROMPTS.map((prompt, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(prompt)}
                  className="quick-prompt-btn"
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg, index) => (
              <div key={msg.id} className={`message-wrapper ${msg.role}`}>
                <div className={`avatar ${msg.role}`}>
                  {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                </div>

                <div>
                  <div className={`message-bubble ${msg.role} ${msg.error ? 'error' : ''}`}>
                    <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>

                    {/* Sources section for assistant answers */}
                    {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
                      <div className="sources-container">
                        <button
                          onClick={() => toggleSources(index)}
                          className="sources-toggle"
                        >
                          <BookOpen size={13} />
                          <span>
                            {expandedSources[index] ? 'Hide Knowledge Sources' : `Verified Sources (${msg.sources.length})`}
                          </span>
                          {expandedSources[index] ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
                        </button>

                        {expandedSources[index] && (
                          <div className="sources-list">
                            {msg.sources.map((src, sIdx) => (
                              <div key={sIdx} className="source-chip">
                                <span className="source-title">
                                  📄 {src.title || src.source} &bull; {src.section}
                                </span>
                                <span className="source-badge">
                                  {Math.round(src.similarity * 100)}% match
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  <div
                    style={{
                      fontSize: '0.7rem',
                      color: '#94a3b8',
                      marginTop: '3px',
                      textAlign: msg.role === 'user' ? 'right' : 'left'
                    }}
                  >
                    {msg.timestamp}
                  </div>
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="message-wrapper assistant">
                <div className="avatar assistant">
                  <Bot size={18} />
                </div>
                <div className="message-bubble assistant">
                  <div className="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      {/* Input Form */}
      <footer className="chat-input-area">
        <form onSubmit={handleSubmit} className="input-form">
          <input
            type="text"
            className="chat-input"
            placeholder="Ask about LPUNEST, UMS, 75% attendance, fee installments, hostel leave, Wi-Fi..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
          />
          <button
            type="submit"
            className="btn-send"
            disabled={!input.trim() || isLoading}
            title="Send Message"
          >
            <Send size={18} />
          </button>
        </form>

        <div className="input-footer">
          <ShieldCheck size={14} className="text-emerald-600" />
          <span>
            Grounded responses powered by OpenRouter LLM and ChromaDB Vector Store.
          </span>
        </div>
      </footer>
    </div>
  )
}
