import { useCallback, useEffect, useState } from "react";
import { api, errorText, type Audit, type Hit, type PieceDetail, type Segment } from "./api";
import { STATE_LABEL, VERDICT_LABEL } from "./App";
import SourcePanel from "./Evidence";

// Standalone lines (greeting, hook, call to action, sign-off) start a new paragraph in the proof.
const STANDALONE = ["greeting", "framing", "cta", "read_next", "sign_off"];

export default function PieceView({ id, onChanged }: { id: number; onChanged: () => void }) {
  const [piece, setPiece] = useState<PieceDetail | null>(null);
  const [message, setMessage] = useState<{ kind: "error" | "ok"; text: string } | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [rejecting, setRejecting] = useState<string | null>(null);

  const load = useCallback(async () => {
    const { data } = await api.GET("/api/pieces/{piece_id}", { params: { path: { piece_id: id } } });
    if (data) setPiece(data);
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  // Checks run in the background (about a minute with the real judge): poll until done.
  useEffect(() => {
    if (piece?.state !== "checks") return;
    const timer = setInterval(async () => {
      await load();
      onChanged();
    }, 2500);
    return () => clearInterval(timer);
  }, [piece?.state, load, onChanged]);

  async function act(run: () => Promise<{ data?: PieceDetail; error?: unknown }>, ok?: string) {
    setMessage(null);
    const { data, error } = await run();
    if (error) {
      setMessage({ kind: "error", text: errorText(error) });
      await load(); // a refused approval is still recorded in the history
    } else if (data) {
      setPiece(data);
      if (ok) setMessage({ kind: "ok", text: ok });
    }
    onChanged();
  }

  const path = { params: { path: { piece_id: id } } };
  const runChecks = () => act(() => api.POST("/api/pieces/{piece_id}/submit", path));
  const approve = () => act(() => api.POST("/api/pieces/{piece_id}/approve", path), "Approved. Schedule it from Campaigns.");
  const reject = (reason: string) =>
    act(() => api.POST("/api/pieces/{piece_id}/reject", { ...path, body: { reason } }), "Rejected.")
      .then(() => setRejecting(null));
  const saveEdit = (text: string) =>
    act(() => api.PUT("/api/pieces/{piece_id}/text", { ...path, body: { text } }),
      "Saved as a new version. Run checks again before approving.")
      .then(() => setEditing(null));
  const override = (hit: Hit, reason: string) =>
    act(() => api.POST("/api/pieces/{piece_id}/overrides", { ...path, body: { hit_index: hit.index, reason } }),
      `Override recorded for ${hit.rule}.`);

  if (!piece) return <p className="muted">Loading…</p>;
  const check = piece.check?.status === "done" ? piece.check : null;
  const hitsFor = (i: number | null) => check?.hits.filter((h) => h.segment === i) ?? [];
  const reviewing = piece.state === "in_review";
  const canEdit = ["draft", "in_review", "rejected", "approved"].includes(piece.state);
  const blockers = piece.blockers.length;

  return (
    <article>
      <header className="piece-head">
        <div>
          <h1>{piece.title}</h1>
          <p className="about">{describePiece(piece)}</p>
          <div className="status">
            <span className="state">{STATE_LABEL[piece.state] ?? piece.state}</span>
            {piece.verdict && <span className={`verdict ${piece.verdict}`}>{VERDICT_LABEL[piece.verdict]}</span>}
            {reviewing && blockers > 0 && (
              <span className="small muted">
                {blockers === 1 ? "1 blocking issue needs" : `${blockers} blocking issues need`} an override with a reason
              </span>
            )}
          </div>
        </div>
        <div className="piece-actions">
          {canEdit && editing === null && <button className="quiet" onClick={() => setEditing(piece.text)}>Edit text</button>}
          {reviewing && <button onClick={() => setRejecting("")}>Reject…</button>}
          {piece.state === "draft" && <button className="primary" onClick={runChecks}>Run checks</button>}
          {reviewing && <button className="primary" onClick={approve}>Approve</button>}
          {piece.state === "checks" && <span className="muted">Checking every sentence…</span>}
        </div>
      </header>

      <div className="piece-notices">
        {message && <p className={`notice ${message.kind}`}>{message.text}</p>}
        {piece.check?.status === "failed" && <p className="notice error">Checks failed: {piece.check.error}. Run them again.</p>}
        {problems(piece).length > 0 && (
          <p className="notice warn">The generator's draft still had problems: {problems(piece).join("; ")}</p>
        )}
        {rejecting !== null && (
          <ReasonForm label="Why reject this piece?" submitLabel="Reject" value={rejecting} onChange={setRejecting}
            onSubmit={reject} onCancel={() => setRejecting(null)} />
        )}
        {editing !== null && (
          <div className="editor">
            <textarea rows={12} value={editing} onChange={(e) => setEditing(e.target.value)} aria-label="Piece text" />
            <p className="small muted">Saving makes a new version. It goes back to draft and has to be checked again.</p>
            <div className="row">
              <button className="primary" disabled={!editing.trim()} onClick={() => saveEdit(editing)}>Save new version</button>
              <button onClick={() => setEditing(null)}>Cancel</button>
            </div>
          </div>
        )}
      </div>

      {!check && <div className="draft-text">{piece.text}</div>}

      {check && hitsFor(null).length > 0 && (
        <div className="whole">
          <div className="margin">
            {hitsFor(null).map((h) => <Note key={h.index} hit={h} reviewing={reviewing} onOverride={override} />)}
          </div>
        </div>
      )}

      {check && (
        <div className="proof" aria-label="Checked text">
          {check.segments.map((seg) => (
            <ProofRow key={seg.index} seg={seg} hits={hitsFor(seg.index)} selected={seg.index === selected}
              reviewing={reviewing} onSelect={() => setSelected(seg.index === selected ? null : seg.index)}
              onOverride={override} />
          ))}
        </div>
      )}

      <History audit={piece.audit} />
    </article>
  );
}

function ProofRow({ seg, hits, selected, reviewing, onSelect, onOverride }: {
  seg: Segment; hits: Hit[]; selected: boolean; reviewing: boolean;
  onSelect: () => void; onOverride: (h: Hit, reason: string) => Promise<void>;
}) {
  const exempt = seg.kind === "exempt";
  const mark = seg.verdict !== "pass" ? seg.verdict : exempt ? "exempt" : "pass";
  const standalone = exempt && STANDALONE.includes(seg.exempt_type ?? "");
  return (
    <div className={`proof-row${standalone ? " break" : ""}`}>
      <div className="rail" aria-hidden="true">
        <span className={`mark ${mark}`}><span className="pt" /></span>
      </div>
      <button className={`sentence${exempt ? " exempt" : ""}`} aria-pressed={selected} onClick={onSelect}
        title={exempt ? `No citation needed (${seg.exempt_type?.replace("_", " ")})` : "Show sources"}>
        {seg.text}
        {seg.citations.length > 0 && <sup>{seg.citations.length}</sup>}
      </button>
      <div className="margin">
        {hits.map((h) => <Note key={h.index} hit={h} reviewing={reviewing} onOverride={onOverride} />)}
        {selected && <SourcePanel segment={seg} hits={hits} />}
      </div>
    </div>
  );
}

function Note({ hit, reviewing, onOverride }: {
  hit: Hit; reviewing: boolean; onOverride: (h: Hit, reason: string) => Promise<void>;
}) {
  const [reason, setReason] = useState<string | null>(null);
  return (
    <div className={`note ${hit.severity}`}>
      <span className="rule">{hit.rule}</span>
      {hit.reason}
      {hit.override_reason && <p className="overridden small">Overridden: {hit.override_reason}</p>}
      {reviewing && hit.severity === "block" && !hit.override_reason && reason === null && (
        <div><button className="quiet small" onClick={() => setReason("")}>Override with a reason</button></div>
      )}
      {reason !== null && (
        <ReasonForm label="Why is the checker wrong here?" submitLabel="Record override" value={reason}
          onChange={setReason} onSubmit={(r) => onOverride(hit, r).then(() => setReason(null))}
          onCancel={() => setReason(null)} />
      )}
    </div>
  );
}

function ReasonForm({ label, value, onChange, onSubmit, onCancel, submitLabel }: {
  label: string; value: string; onChange: (v: string) => void;
  onSubmit: (v: string) => void; onCancel: () => void; submitLabel: string;
}) {
  return (
    <form onSubmit={(e) => { e.preventDefault(); onSubmit(value); }}>
      <label className="field">{label}
        <textarea rows={2} value={value} onChange={(e) => onChange(e.target.value)} autoFocus />
      </label>
      <div className="row">
        <button className="primary" type="submit" disabled={!value.trim()}>{submitLabel}</button>
        <button type="button" onClick={onCancel}>Cancel</button>
      </div>
    </form>
  );
}

function History({ audit }: { audit: Audit[] }) {
  if (!audit.length) return null;
  return (
    <details className="history">
      <summary>History ({audit.length} {audit.length === 1 ? "event" : "events"})</summary>
      <ol>
        {audit.map((a, i) => (
          <li key={i}>
            <time dateTime={a.at}>{new Date(a.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time>
            <span>{describeEvent(a)}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}

function describePiece(p: PieceDetail): string {
  const kind = p.kind === "explainer_summary" ? "Explainer summary" : "Newsletter";
  const parts = [`${kind} for ${p.segment} readers, version ${p.version}.`];
  const g = p.generation as { generator_model?: string; retrieved_chunks?: string[]; corpus_version?: string } | null;
  if (p.source === "generated" && g) {
    parts.push(`Written by ${g.generator_model} from ${g.retrieved_chunks?.length ?? 0} passages of ${g.corpus_version}.`);
  }
  if (p.check?.status === "done") parts.push(`Checked by ${p.check.judge_model} against ${p.check.corpus_version}.`);
  return parts.join(" ");
}

function problems(p: PieceDetail): string[] {
  return p.source === "generated" ? ((p.generation as { problems?: string[] } | null)?.problems ?? []) : [];
}

function describeEvent(a: Audit): string {
  const who = a.actor === "system" ? "The checker" : "You";
  const d = (a.detail ?? {}) as Record<string, unknown>;
  switch (a.action) {
    case "create": return d.source === "generated" ? "Draft generated." : "Draft created.";
    case "submit": return `${who} started checks on version ${d.version}.`;
    case "checks_done": return `Checks finished: ${VERDICT_LABEL[String(d.verdict)] ?? d.verdict}${(d.rules as string[])?.length ? ` (${(d.rules as string[]).join(", ")})` : ""}.`;
    case "checks_failed": return `Checks failed: ${d.error}.`;
    case "approve_refused": return `Approval refused: ${(d.rules as string[])?.join(", ")} still blocking.`;
    case "override": return `${who} overrode ${d.rule}: "${d.reason}".`;
    case "approve": return `${who} approved it.`;
    case "reject": return `${who} rejected it: "${d.reason}".`;
    case "edit": return `${who} saved version ${d.version}.`;
    case "schedule": return `${who} scheduled it (${(d.channels as string[])?.join(" and ")}).`;
    case "cancel": return `${who} cancelled the campaign.`;
    case "send": return "Sent.";
    case "send_failed": return "Sending failed: nothing was delivered.";
    case "generation_problems": return "The generator's draft still had problems.";
    default: return a.action.replace("_", " ");
  }
}
