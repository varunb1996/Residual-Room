import { useCallback, useEffect, useState } from "react";
import { api, errorText, type PieceSummary } from "./api";
import type { components } from "./schema";

type Campaign = components["schemas"]["CampaignOut"];
type Preview = components["schemas"]["PreviewOut"];
type Segment = "transitioning" | "practitioner";
type Channel = "email" | "telegram" | "whatsapp";

const SEGMENTS: Segment[] = ["practitioner", "transitioning"];
const CHANNELS: Channel[] = ["email", "telegram", "whatsapp"];

export default function Campaigns() {
  const [pieces, setPieces] = useState<PieceSummary[]>([]);
  const [campaigns, setCampaigns] = useState<Campaign[]>([]);
  const [pieceId, setPieceId] = useState<number | null>(null);
  const [segments, setSegments] = useState<Segment[]>(["practitioner"]);
  const [channels, setChannels] = useState<Channel[]>(["email", "telegram"]);
  const [when, setWhen] = useState("");
  const [preview, setPreview] = useState<Preview | null>(null);
  const [message, setMessage] = useState<{ kind: "ok" | "error"; text: string } | null>(null);

  const refresh = useCallback(async () => {
    const [p, c] = await Promise.all([api.GET("/api/pieces"), api.GET("/api/campaigns")]);
    if (p.data) setPieces(p.data);
    if (c.data) setCampaigns(c.data);
  }, []);

  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, 5000); // deliveries and clicks update as they happen
    return () => clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    setPreview(null);
    if (pieceId === null) return;
    api.GET("/api/pieces/{piece_id}/preview", { params: { path: { piece_id: pieceId } } })
      .then(({ data }) => data && setPreview(data));
  }, [pieceId]);

  const approved = pieces.filter((p) => p.state === "approved");
  const toggle = <T,>(list: T[], item: T) => (list.includes(item) ? list.filter((x) => x !== item) : [...list, item]);

  async function schedule() {
    if (pieceId === null) return;
    setMessage(null);
    const { error } = await api.POST("/api/campaigns", {
      body: { piece_id: pieceId, segments, channels, scheduled_at: when ? new Date(when).toISOString() : null },
    });
    if (error) setMessage({ kind: "error", text: errorText(error) });
    else {
      setMessage({ kind: "ok", text: when ? "Scheduled." : "Sending now…" });
      setPieceId(null);
    }
    refresh();
  }

  async function cancel(id: number) {
    const { error } = await api.POST("/api/campaigns/{campaign_id}/cancel", { params: { path: { campaign_id: id } } });
    if (error) setMessage({ kind: "error", text: errorText(error) });
    refresh();
  }

  return (
    <div className="piece">
      <section className="card">
        <h1>New campaign</h1>
        <p className="muted small">Only approved pieces can be scheduled. Email goes only to TEST_EMAILS; Telegram only to the test channel; WhatsApp is a mock.</p>
        <label>
          Approved piece
          <select value={pieceId ?? ""} onChange={(e) => setPieceId(e.target.value ? Number(e.target.value) : null)}>
            <option value="">{approved.length ? "Choose…" : "No approved pieces yet"}</option>
            {approved.map((p) => <option key={p.id} value={p.id}>#{p.id} · {p.title}</option>)}
          </select>
        </label>
        <div className="row">
          <fieldset>
            <legend>Segments</legend>
            {SEGMENTS.map((s) => (
              <label key={s} className="check">
                <input type="checkbox" checked={segments.includes(s)} onChange={() => setSegments(toggle(segments, s))} /> {s}
              </label>
            ))}
          </fieldset>
          <fieldset>
            <legend>Channels</legend>
            {CHANNELS.map((c) => (
              <label key={c} className="check">
                <input type="checkbox" checked={channels.includes(c)} onChange={() => setChannels(toggle(channels, c))} /> {c}
              </label>
            ))}
          </fieldset>
        </div>
        <label>
          Send at (leave empty to send now)
          <input type="datetime-local" value={when} onChange={(e) => setWhen(e.target.value)} />
        </label>
        {message && <p className={message.kind === "error" ? "error" : "ok"}>{message.text}</p>}
        <div className="actions">
          <button className="primary" disabled={pieceId === null || !segments.length || !channels.length} onClick={schedule}>
            {when ? "Schedule" : "Send now"}
          </button>
        </div>
      </section>

      {preview && (
        <section className="card">
          <h2>Preview</h2>
          <h3>Email: {preview.email_subject}</h3>
          {/* Sandboxed: the email HTML is shown, never run. */}
          <iframe title="email preview" className="email-preview" sandbox="" srcDoc={preview.email_html} />
          <h3>Telegram</h3>
          <pre className="raw-preview">{preview.telegram_html}</pre>
          <h3>WhatsApp (template)</h3>
          <pre className="raw-preview">{preview.whatsapp_text}</pre>
        </section>
      )}

      <section className="card">
        <h2>Campaigns</h2>
        {campaigns.length === 0 && <p className="muted small">None yet.</p>}
        {campaigns.map((c) => (
          <div key={c.id} className="campaign">
            <div className="piece-head">
              <div>
                <strong>#{c.id} · {c.piece_title}</strong>
                <p className="muted small">
                  {c.segments.join(", ")} · {c.channels.join(", ")} · {c.status === "scheduled" ? "at" : "scheduled"}{" "}
                  {new Date(c.scheduled_at).toLocaleString()}
                </p>
              </div>
              <div className="badges">
                <span className={`badge state-${c.status === "sent" ? "approved" : c.status === "failed" ? "rejected" : "in_review"}`}>{c.status}</span>
                {c.status === "scheduled" && <button className="small-btn" onClick={() => cancel(c.id)}>Cancel</button>}
              </div>
            </div>
            {c.deliveries.length > 0 && (
              <table className="audit">
                <thead><tr><th>Channel</th><th>Segment</th><th>Recipient</th><th>Status</th><th>Detail</th></tr></thead>
                <tbody>
                  {c.deliveries.map((d, i) => (
                    <tr key={i}>
                      <td>{d.channel}</td><td>{d.segment}</td><td>{d.recipient}</td>
                      <td><span className={`badge verdict-${d.status === "sent" ? "pass" : d.status === "refused" ? "warn" : "block"}`}>{d.status}</span></td>
                      <td className="small">{d.error ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            <p className="small">
              <span className="muted">Clicks: </span>
              {c.clicks.length === 0 ? "none yet" : c.clicks.map((k) => `${k.channel}/${k.segment} ${k.label}: ${k.clicks}`).join(" · ")}
            </p>
          </div>
        ))}
      </section>
    </div>
  );
}
