import { useState } from "react";
import { api, errorText } from "./api";

// Until generation exists (Prompt 3), pieces are created from pasted text.
export default function NewPiece({ onCreated, onCancel }: { onCreated: (id: number) => void; onCancel: () => void }) {
  const [title, setTitle] = useState("");
  const [segment, setSegment] = useState<"practitioner" | "transitioning">("practitioner");
  const [issueDate, setIssueDate] = useState("");
  const [text, setText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function create() {
    setBusy(true);
    setError(null);
    const { data, error } = await api.POST("/api/pieces", {
      body: { title, segment, text, kind: "newsletter", issue_date: issueDate || null },
    });
    setBusy(false);
    if (error) setError(errorText(error));
    else if (data) onCreated(data.id);
  }

  return (
    <section className="card">
      <h1>New piece</h1>
      <p className="muted small">Paste a draft. Generation will create pieces here once it's built.</p>
      <label>
        Title
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Issue 2: why your judge flips" />
      </label>
      <div className="row">
        <label>
          Reader segment
          <select value={segment} onChange={(e) => setSegment(e.target.value as typeof segment)}>
            <option value="practitioner">practitioner</option>
            <option value="transitioning">transitioning</option>
          </select>
        </label>
        <label>
          Issue date (optional)
          <input type="date" value={issueDate} onChange={(e) => setIssueDate(e.target.value)} />
        </label>
      </div>
      <label>
        Text
        <textarea rows={10} value={text} onChange={(e) => setText(e.target.value)}
          placeholder="Use [cite:source-id] after a claim to cite it." />
      </label>
      {error && <p className="error">{error}</p>}
      <div className="actions">
        <button className="primary" disabled={busy || !title.trim() || !text.trim()} onClick={create}>
          Create draft
        </button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    </section>
  );
}
