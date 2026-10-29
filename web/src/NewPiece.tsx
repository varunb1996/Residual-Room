import { useState } from "react";
import { api, errorText } from "./api";

// A hand-written draft goes through exactly the same checks as a generated one.
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
    <section className="sheet">
      <h1>Paste a draft</h1>
      <p className="lede">Write [cite:source-id] after each factual sentence. Sentences without a citation are flagged when you run checks.</p>
      <label className="field">
        Title
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Issue 2: why your judge flips" />
      </label>
      <div className="row">
        <label className="field grow">
          Readers
          <select value={segment} onChange={(e) => setSegment(e.target.value as typeof segment)}>
            <option value="practitioner">Practitioners</option>
            <option value="transitioning">Engineers moving into AI</option>
          </select>
        </label>
        <label className="field grow">
          Issue date (optional)
          <input type="date" value={issueDate} onChange={(e) => setIssueDate(e.target.value)} />
        </label>
      </div>
      <label className="field">
        Text
        <textarea className="draft-input" rows={12} value={text} onChange={(e) => setText(e.target.value)} />
      </label>
      {error && <p className="notice error">{error}</p>}
      <div className="row">
        <button className="primary" disabled={busy || !title.trim() || !text.trim()} onClick={create}>
          Save draft
        </button>
        <button onClick={onCancel}>Cancel</button>
      </div>
    </section>
  );
}
