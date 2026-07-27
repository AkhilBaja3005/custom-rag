"use client";

import React, { useState, useEffect, useRef } from "react";
import { 
  Zap, 
  BookOpen, 
  UploadCloud, 
  Trash2, 
  Send, 
  Cpu, 
  Database, 
  FileText, 
  CheckCircle2, 
  Sparkles, 
  ShieldCheck, 
  Layers, 
  RefreshCw
} from "lucide-react";

interface Collection {
  name: string;
  points_count: number;
}

interface Source {
  page: number;
  text: string;
  rerank_score?: number;
  rrf_score?: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  isCacheHit?: boolean;
}

export default function Home() {
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

  // State
  const [collections, setCollections] = useState<Collection[]>([]);
  const [selectedCollection, setSelectedCollection] = useState<string>("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputQuery, setInputQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedSource, setSelectedSource] = useState<Source | null>(null);

  // Ingestion State
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [customColName, setCustomColName] = useState("");
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const [ingestProgress, setIngestProgress] = useState<number>(0);
  const [isUploading, setIsUploading] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Fetch Collections from FastAPI
  const fetchCollections = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/collections`);
      if (res.ok) {
        const data = await res.json();
        setCollections(data.collections || []);
        if (data.collections?.length > 0 && !selectedCollection) {
          setSelectedCollection(data.collections[0].name);
        }
      }
    } catch (e) {
      console.error("Backend microservice API offline. Start api.py backend on port 8000.", e);
    }
  };

  useEffect(() => {
    fetchCollections();
  }, []);

  // Delete Collection
  const handleDeleteCollection = async (colName: string) => {
    if (!confirm(`Are you sure you want to delete collection '${colName}'?`)) return;
    try {
      const res = await fetch(`${API_BASE}/api/collections/${colName}`, { method: "DELETE" });
      if (res.ok) {
        if (selectedCollection === colName) {
          setSelectedCollection("");
          setMessages([]);
        }
        fetchCollections();
      }
    } catch {
      alert("Failed to delete collection.");
    }
  };

  // Upload & Start Streaming Ingestion Job
  const handleUploadPDF = async () => {
    if (!uploadFile) return;
    setIsUploading(true);
    setIngestStatus("Uploading PDF file...");
    setIngestProgress(5);

    const formData = new FormData();
    formData.append("file", uploadFile);
    if (customColName.trim()) {
      formData.append("collection_name", customColName.trim());
    }

    try {
      const res = await fetch(`${API_BASE}/api/ingest`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");
      const data = await res.json();
      const jobId = data.job_id;
      const targetCol = data.collection_name;

      // Poll Ingestion Status
      const interval = setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_BASE}/api/ingest/status/${jobId}`);
          if (statusRes.ok) {
            const statusData = await statusRes.json();
            setIngestProgress(statusData.progress || 10);
            setIngestStatus(statusData.details || "Indexing PDF chunks...");

            if (statusData.status === "completed") {
              clearInterval(interval);
              setIsUploading(false);
              setIngestStatus("✅ Ingestion Completed!");
              setUploadFile(null);
              setCustomColName("");
              setSelectedCollection(targetCol);
              setMessages([]);
              fetchCollections();
            } else if (statusData.status === "failed") {
              clearInterval(interval);
              setIsUploading(false);
              setIngestStatus(`❌ Failed: ${statusData.error}`);
            }
          }
        } catch {
          clearInterval(interval);
          setIsUploading(false);
        }
      }, 1000);
    } catch {
      setIsUploading(false);
      setIngestStatus("❌ Error connecting to backend microservice.");
    }
  };

  // Live SSE Stream Query Execution
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputQuery.trim() || !selectedCollection || isLoading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputQuery,
    };

    setMessages((prev) => [...prev, userMsg]);
    const currentQuery = inputQuery;
    setInputQuery("");
    setIsLoading(true);

    const assistantMsgId = (Date.now() + 1).toString();
    let assistantContent = "";
    let sources: Source[] = [];
    let isCacheHit = false;

    // Add empty assistant placeholder
    setMessages((prev) => [
      ...prev,
      { id: assistantMsgId, role: "assistant", content: "", sources: [] },
    ]);

    try {
      const sseUrl = `${API_BASE}/api/query/stream?query=${encodeURIComponent(currentQuery)}&collection_name=${encodeURIComponent(selectedCollection)}`;
      const eventSource = new EventSource(sseUrl);

      eventSource.addEventListener("metadata", (event) => {
        const data = JSON.parse(event.data);
        sources = data.sources || [];
        isCacheHit = data.is_cache_hit || false;

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, sources, isCacheHit }
              : msg
          )
        );
      });

      eventSource.addEventListener("token", (event) => {
        const data = JSON.parse(event.data);
        assistantContent += data.content;

        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMsgId
              ? { ...msg, content: assistantContent }
              : msg
          )
        );
      });

      eventSource.addEventListener("done", () => {
        eventSource.close();
        setIsLoading(false);
      });

      eventSource.addEventListener("error", () => {
        eventSource.close();
        setIsLoading(false);
      });
    } catch {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 font-sans antialiased overflow-hidden">
      {/* ------------------------------------------------------------- */}
      {/* LEFT SIDEBAR: COLLECTION MANAGER & INGESTION CONTROL CENTER    */}
      {/* ------------------------------------------------------------- */}
      <aside className="w-80 bg-zinc-900/60 border-r border-zinc-800/80 flex flex-col justify-between backdrop-blur-xl p-5 select-none">
        <div className="space-y-6 overflow-y-auto pr-1">
          {/* Header */}
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-br from-purple-600 to-cyan-600 rounded-xl shadow-lg shadow-purple-900/30">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg tracking-tight bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                SOTA RAG Studio
              </h1>
              <p className="text-xs text-zinc-400 font-mono">10k-Page Enterprise Platform</p>
            </div>
          </div>

          {/* Hardware & System Status */}
          <div className="bg-zinc-900/80 border border-zinc-800 rounded-xl p-3.5 space-y-2">
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span className="flex items-center gap-1.5 font-medium"><Cpu className="w-3.5 h-3.5 text-cyan-400" /> Accel Device</span>
              <span className="px-2 py-0.5 bg-cyan-950 text-cyan-400 border border-cyan-800/50 rounded-md font-mono text-[11px]">MPS / CUDA</span>
            </div>
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span className="flex items-center gap-1.5 font-medium"><Layers className="w-3.5 h-3.5 text-purple-400" /> Vector HNSW</span>
              <span className="px-2 py-0.5 bg-purple-950 text-purple-400 border border-purple-800/50 rounded-md font-mono text-[11px]">m=16 ef=100</span>
            </div>
            <div className="flex items-center justify-between text-xs text-zinc-400">
              <span className="flex items-center gap-1.5 font-medium"><ShieldCheck className="w-3.5 h-3.5 text-emerald-400" /> Groundedness</span>
              <span className="text-emerald-400 font-mono text-[11px]">100% Zero-Extrap</span>
            </div>
          </div>

          {/* Document Collection Selector */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-purple-400" /> Indexed Collections
              </label>
              <button 
                onClick={fetchCollections} 
                className="text-zinc-500 hover:text-zinc-300 transition-colors"
                title="Refresh Collections"
              >
                <RefreshCw className="w-3.5 h-3.5" />
              </button>
            </div>

            {collections.length === 0 ? (
              <div className="p-3 bg-zinc-900/40 border border-dashed border-zinc-800 rounded-xl text-center">
                <p className="text-xs text-zinc-500">No collections found. Upload a PDF to start.</p>
              </div>
            ) : (
              <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                {collections.map((col) => (
                  <div
                    key={col.name}
                    onClick={() => {
                      setSelectedCollection(col.name);
                      setMessages([]);
                    }}
                    className={`flex items-center justify-between p-2.5 rounded-xl text-xs font-medium cursor-pointer transition-all border ${
                      selectedCollection === col.name
                        ? "bg-purple-950/60 border-purple-500/50 text-purple-200 shadow-md shadow-purple-950/40"
                        : "bg-zinc-900/40 border-zinc-800/60 text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200"
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <BookOpen className="w-3.5 h-3.5 text-purple-400 shrink-0" />
                      <span className="truncate">{col.name}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="px-1.5 py-0.5 bg-zinc-800/80 text-zinc-400 rounded text-[10px] font-mono">
                        {col.points_count.toLocaleString()} pts
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteCollection(col.name);
                        }}
                        className="text-zinc-500 hover:text-red-400 transition-colors p-0.5"
                        title="Delete Collection"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* PDF Upload & Asynchronous Ingestion */}
          <div className="space-y-3 pt-2">
            <label className="text-xs font-semibold uppercase tracking-wider text-zinc-400 flex items-center gap-1.5">
              <UploadCloud className="w-3.5 h-3.5 text-cyan-400" /> Ingest PDF Document
            </label>

            <input
              type="text"
              placeholder="Custom Collection Name (Optional)"
              value={customColName}
              onChange={(e) => setCustomColName(e.target.value)}
              className="w-full bg-zinc-900 border border-zinc-800 rounded-xl px-3 py-2 text-xs text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-purple-500 transition-colors"
            />

            <label className="flex flex-col items-center justify-center border border-dashed border-zinc-800 hover:border-purple-500/50 rounded-xl p-4 cursor-pointer bg-zinc-900/30 hover:bg-zinc-900/60 transition-all">
              <UploadCloud className="w-6 h-6 text-zinc-500 mb-1" />
              <span className="text-xs text-zinc-300 font-medium">
                {uploadFile ? uploadFile.name : "Select PDF (up to 10k pages)"}
              </span>
              <span className="text-[10px] text-zinc-500 mt-0.5">Max file size 1,000 MB</span>
              <input
                type="file"
                accept="application/pdf"
                className="hidden"
                onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
              />
            </label>

            {uploadFile && (
              <button
                onClick={handleUploadPDF}
                disabled={isUploading}
                className="w-full bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 text-white font-medium py-2 px-4 rounded-xl text-xs shadow-lg shadow-purple-900/20 transition-all disabled:opacity-50 flex items-center justify-center gap-1.5"
              >
                {isUploading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
                {isUploading ? "Streaming Ingestion..." : "Start Ingestion Job"}
              </button>
            )}

            {ingestStatus && (
              <div className="p-3 bg-zinc-900 border border-zinc-800 rounded-xl space-y-2">
                <div className="flex justify-between items-center text-[11px] text-zinc-400">
                  <span className="truncate">{ingestStatus}</span>
                  <span className="font-mono font-bold text-cyan-400">{ingestProgress}%</span>
                </div>
                <div className="w-full bg-zinc-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className="bg-gradient-to-r from-purple-500 to-cyan-500 h-full transition-all duration-300"
                    style={{ width: `${ingestProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="pt-4 border-t border-zinc-800/80 text-[11px] text-zinc-500 flex justify-between items-center">
          <span>FastAPI SSE + Next.js 15</span>
          <span className="flex items-center gap-1 text-emerald-400 font-mono"><CheckCircle2 className="w-3 h-3" /> Online</span>
        </div>
      </aside>

      {/* ------------------------------------------------------------- */}
      {/* MAIN CHAT WORKSPACE (SPLIT SCREEN LAYOUT)                      */}
      {/* ------------------------------------------------------------- */}
      <main className="flex-1 flex flex-col h-full bg-zinc-950 relative overflow-hidden">
        {/* Top Header Navbar */}
        <header className="h-16 border-b border-zinc-800/80 bg-zinc-900/40 backdrop-blur-md px-6 flex items-center justify-between select-none shrink-0">
          <div className="flex items-center gap-3">
            <FileText className="w-5 h-5 text-purple-400" />
            <div>
              <h2 className="font-semibold text-sm text-zinc-200">
                {selectedCollection ? `Collection: ${selectedCollection}` : "No Collection Selected"}
              </h2>
              <p className="text-[11px] text-zinc-500">HyDE Expansion • BM25+Dense RRF • Cross-Encoder Rerank</p>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs">
            <span className="px-3 py-1 bg-zinc-900 border border-zinc-800 text-zinc-400 rounded-full font-mono text-[11px] flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" /> SSE Real-Time Stream
            </span>
          </div>
        </header>

        {/* Split Screen Workspace */}
        <div className="flex-1 flex overflow-hidden">
          {/* Chat Conversation View */}
          <div className="flex-1 flex flex-col h-full relative">
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-4">
                  <div className="p-4 bg-purple-950/30 border border-purple-500/20 rounded-2xl">
                    <Sparkles className="w-8 h-8 text-purple-400" />
                  </div>
                  <div>
                    <h3 className="text-base font-semibold text-zinc-200">Enter your query to start searching</h3>
                    <p className="text-xs text-zinc-500 max-w-sm mt-1">
                      Querying collection <code className="text-purple-400 font-mono">{selectedCollection || "none"}</code> using SOTA zero-extrapolation answer synthesis.
                    </p>
                  </div>
                </div>
              ) : (
                messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`flex flex-col space-y-2 ${
                      msg.role === "user" ? "items-end" : "items-start"
                    }`}
                  >
                    <div
                      className={`max-w-3xl rounded-2xl px-5 py-3.5 text-sm leading-relaxed ${
                        msg.role === "user"
                          ? "bg-gradient-to-r from-purple-600 to-cyan-600 text-white rounded-br-none shadow-lg shadow-purple-950/30"
                          : "bg-zinc-900 border border-zinc-800 text-zinc-200 rounded-bl-none"
                      }`}
                    >
                      {msg.isCacheHit && (
                        <div className="inline-flex items-center gap-1 px-2 py-0.5 mb-2 bg-cyan-950/80 border border-cyan-800 text-cyan-400 rounded-md text-[10px] font-mono font-semibold">
                          <Zap className="w-3 h-3 text-cyan-400" /> Sub-10ms Semantic Vector Cache Hit
                        </div>
                      )}

                      <p className="whitespace-pre-wrap">{msg.content || (isLoading && msg.role === "assistant" ? "Thinking & searching vector graph..." : "")}</p>

                      {/* Source Citation Chips */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-zinc-800/80 space-y-2">
                          <span className="text-[11px] font-semibold text-zinc-400 uppercase tracking-wider block">
                            Retrieved Context Sources:
                          </span>
                          <div className="flex flex-wrap gap-2">
                            {msg.sources.map((src, sIdx) => (
                              <button
                                key={sIdx}
                                onClick={() => setSelectedSource(src)}
                                className={`text-[11px] font-mono px-2.5 py-1 rounded-lg border transition-all flex items-center gap-1.5 ${
                                  selectedSource?.text === src.text
                                    ? "bg-purple-900/60 border-purple-500 text-purple-200"
                                    : "bg-zinc-800/60 border-zinc-700/60 text-zinc-300 hover:bg-zinc-800"
                                }`}
                              >
                                <span>Source: Page {src.page}</span>
                                {src.rerank_score !== undefined && (
                                  <span className="text-[10px] text-cyan-400 font-bold">
                                    ({src.rerank_score.toFixed(2)})
                                  </span>
                                )}
                              </button>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Chat Input Field */}
            <div className="p-4 border-t border-zinc-800/80 bg-zinc-900/40 backdrop-blur-md">
              <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto relative flex items-center">
                <input
                  type="text"
                  placeholder={
                    selectedCollection
                      ? `Ask a question about '${selectedCollection}'...`
                      : "Select or ingest a collection to start..."
                  }
                  disabled={!selectedCollection || isLoading}
                  value={inputQuery}
                  onChange={(e) => setInputQuery(e.target.value)}
                  className="w-full bg-zinc-900 border border-zinc-800 rounded-2xl pl-5 pr-14 py-3.5 text-sm text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-purple-500/80 transition-colors shadow-inner disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={!inputQuery.trim() || !selectedCollection || isLoading}
                  className="absolute right-2 p-2.5 bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 text-white rounded-xl shadow-md transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                >
                  <Send className="w-4 h-4" />
                </button>
              </form>
            </div>
          </div>

          {/* Side-by-Side PDF Context Inspector Panel */}
          {selectedSource && (
            <aside className="w-96 border-l border-zinc-800 bg-zinc-900/50 backdrop-blur-xl p-5 flex flex-col h-full overflow-y-auto animate-in slide-in-from-right-10 duration-200">
              <div className="flex items-center justify-between pb-4 border-b border-zinc-800">
                <div className="flex items-center gap-2">
                  <FileText className="w-4 h-4 text-purple-400" />
                  <h3 className="font-semibold text-sm text-zinc-200">Context Source Inspector</h3>
                </div>
                <button
                  onClick={() => setSelectedSource(null)}
                  className="text-zinc-500 hover:text-zinc-300 text-xs px-2 py-1 bg-zinc-800 rounded-md"
                >
                  Close
                </button>
              </div>

              <div className="space-y-4 mt-4 text-xs">
                <div className="p-3 bg-zinc-900 border border-zinc-800 rounded-xl space-y-1">
                  <div className="flex justify-between text-zinc-400">
                    <span>Document Page:</span>
                    <span className="font-mono text-purple-400 font-bold">Page {selectedSource.page}</span>
                  </div>
                  {selectedSource.rerank_score !== undefined && (
                    <div className="flex justify-between text-zinc-400">
                      <span>Cross-Encoder Score:</span>
                      <span className="font-mono text-cyan-400 font-bold">{selectedSource.rerank_score.toFixed(4)}</span>
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <label className="text-[11px] font-semibold uppercase tracking-wider text-zinc-400">
                    Exact Chunk Content
                  </label>
                  <div className="p-4 bg-zinc-900 border border-zinc-800/80 rounded-xl text-zinc-300 leading-relaxed font-mono whitespace-pre-wrap text-[11px]">
                    {selectedSource.text}
                  </div>
                </div>
              </div>
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}
