/**
 * Account and sessions.
 *
 * The controls here map exactly onto what Alpha's backend can actually do:
 * see the account, see every live session, end one, end all of them, change the
 * password, sign out. Nothing is decorative — each button calls the Alpha
 * function that performs the operation, and the session list is a live query, so
 * a session revoked in another tab disappears from this one.
 *
 * Session lifetime is the account's business, so it is stated plainly rather
 * than hidden: an absolute expiry, an idle expiry, and what a password change
 * does to everything else.
 */

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  EmptyNote,
  Eyebrow,
  Frame,
  KeyValue,
  Mono,
  Pill,
  Rule,
  Stat,
  StatGrid,
  WarningNote,
} from "@/components/alpha/studio";
import { alphaAuthErrorMessage, useAuth } from "@/hooks/use-auth";
import { Loader2, LogOut, MonitorSmartphone, ShieldCheck, TriangleAlert } from "lucide-react";
import { useState } from "react";

function formatTimestamp(ms: number | null | undefined): string {
  if (!ms) return "—";
  return new Date(ms).toLocaleString();
}

function formatRemaining(expiresAt: number): string {
  const remaining = expiresAt - Date.now();
  if (remaining <= 0) return "expired";
  const days = Math.floor(remaining / 86_400_000);
  const hours = Math.floor((remaining % 86_400_000) / 3_600_000);
  if (days > 0) return `${days}d ${hours}h`;
  const minutes = Math.floor((remaining % 3_600_000) / 60_000);
  return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
}

export function AccountPanel() {
  const {
    user,
    session,
    sessions,
    signOut,
    signOutEverywhere,
    revokeSession,
    changePassword,
  } = useAuth();

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleChangePassword = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setNotice(null);
    setError(null);
    if (newPassword !== confirmPassword) {
      setError("The new password and its confirmation do not match.");
      return;
    }
    setBusy("password");
    try {
      await changePassword({ currentPassword, newPassword });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setNotice(
        "Password changed. Every other session was ended; this device received a new session.",
      );
    } catch (caught) {
      setError(alphaAuthErrorMessage(caught));
    } finally {
      setBusy(null);
    }
  };

  const handleRevoke = async (sessionId: string) => {
    setError(null);
    setBusy(sessionId);
    try {
      await revokeSession(sessionId);
      setNotice("Session ended.");
    } catch (caught) {
      setError(alphaAuthErrorMessage(caught));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="space-y-6">
      <Frame
        title="Account"
        status="ready"
        lede="Alpha's own account record. Identity is resolved from the session on every request; no provider holds this account."
        actions={
          <Button variant="outline" size="sm" onClick={() => void signOut()}>
            <LogOut className="mr-2 size-3.5" />
            Sign out
          </Button>
        }
      >
        <StatGrid>
          <Stat label="Email" value={<span className="font-mono text-sm">{user?.email ?? "—"}</span>} />
          <Stat label="Display name" value={user?.displayName ?? "—"} />
          <Stat
            label="Role"
            value={<Pill>{user?.role ?? "user"}</Pill>}
            hint="role checks are enforced server-side"
          />
          <Stat
            label="Account state"
            value={<Pill>{user?.status ?? "—"}</Pill>}
            hint={user?.status === "suspended" ? "sign-in is refused" : "sign-in allowed"}
          />
        </StatGrid>
        <div className="mt-5">
          <KeyValue label="Created">{formatTimestamp(user?.createdAt)}</KeyValue>
          <KeyValue label="Last sign-in">{formatTimestamp(user?.lastSignInAt)}</KeyValue>
          <KeyValue label="Account id">
            <span className="font-mono text-[11px]">{user?.id ?? "—"}</span>
          </KeyValue>
          <KeyValue label="Password storage">PBKDF2-HMAC-SHA256, per-account salt</KeyValue>
        </div>
      </Frame>

      <Frame
        title="Sessions"
        status="ready"
        lede="Every device holding a session for this account. Revoking one takes effect immediately — the query behind this list is live."
        actions={
          <Button
            variant="outline"
            size="sm"
            disabled={busy !== null || sessions.length === 0}
            onClick={async () => {
              setNotice(null);
              setError(null);
              setBusy("all");
              try {
                await signOutEverywhere();
              } catch (caught) {
                setError(alphaAuthErrorMessage(caught));
              } finally {
                setBusy(null);
              }
            }}
          >
            <MonitorSmartphone className="mr-2 size-3.5" />
            Sign out everywhere
          </Button>
        }
      >
        <div className="grid gap-4 sm:grid-cols-3">
          <Stat label="This session started" value={formatTimestamp(session?.createdAt)} />
          <Stat label="Expires in" value={session ? formatRemaining(session.expiresAt) : "—"} hint="30-day absolute limit" />
          <Stat label="Last seen" value={formatTimestamp(session?.lastSeenAt)} hint="7-day idle limit" />
        </div>

        <Rule className="my-5" />

        {sessions.length === 0 ? (
          <EmptyNote>No active sessions are recorded for this account.</EmptyNote>
        ) : (
          <div className="space-y-2">
            {sessions.map((entry) => (
              <div
                key={entry.id}
                className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-border px-4 py-3"
              >
                <div className="min-w-0 space-y-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <Eyebrow>{entry.current ? "this device" : "other device"}</Eyebrow>
                    {entry.current ? <Pill>current</Pill> : null}
                  </div>
                  <p className="truncate text-[11px] text-muted-foreground">
                    {entry.userAgent ?? "user agent not recorded"}
                  </p>
                  <p className="text-[11px] text-muted-foreground">
                    started {formatTimestamp(entry.createdAt)} · expires in {formatRemaining(entry.expiresAt)}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  disabled={busy !== null}
                  onClick={() => void handleRevoke(entry.id)}
                >
                  {busy === entry.id ? <Loader2 className="size-3.5 animate-spin" /> : "End session"}
                </Button>
              </div>
            ))}
          </div>
        )}
      </Frame>

      <Frame
        title="Change password"
        status="ready"
        lede="The new password is derived with a fresh random salt. Every session issued before this moment is retired, and this device is given a new one."
      >
        <form onSubmit={handleChangePassword} className="grid gap-4 sm:grid-cols-3">
          <div className="space-y-1.5">
            <Label htmlFor="currentPassword" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
              Current password
            </Label>
            <Input
              id="currentPassword"
              type="password"
              autoComplete="current-password"
              value={currentPassword}
              onChange={(event) => setCurrentPassword(event.target.value)}
              required
              disabled={busy !== null}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="newPassword" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
              New password
            </Label>
            <Input
              id="newPassword"
              type="password"
              autoComplete="new-password"
              value={newPassword}
              onChange={(event) => setNewPassword(event.target.value)}
              required
              disabled={busy !== null}
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="confirmPassword" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
              Confirm new password
            </Label>
            <Input
              id="confirmPassword"
              type="password"
              autoComplete="new-password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              required
              disabled={busy !== null}
            />
          </div>
          <div className="sm:col-span-3">
            <Button type="submit" disabled={busy !== null}>
              {busy === "password" ? <Loader2 className="mr-2 size-3.5 animate-spin" /> : null}
              Change password
            </Button>
          </div>
        </form>

        {notice ? (
          <p className="mt-4 flex items-start gap-2 rounded-md border border-chart-2/40 bg-chart-2/10 px-3 py-2 text-[11px] leading-4 text-foreground">
            <ShieldCheck className="mt-0.5 size-3.5 shrink-0 text-chart-2" />
            <span>{notice}</span>
          </p>
        ) : null}
        {error ? (
          <p className="mt-4 flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-[11px] leading-4 text-foreground">
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-destructive" />
            <span>{error}</span>
          </p>
        ) : null}
      </Frame>

      <WarningNote>
        Alpha talks to its backend through Convex function calls rather than cookies, so the session token is held in
        this browser's <Mono>localStorage</Mono> and is readable by any script on this origin. The protections that do
        apply are on the server: the token is stored only as a hash, sessions expire absolutely and on idle, every
        session can be revoked, and a password change retires all of them. Moving to an httpOnly cookie would need an
        Alpha HTTP auth layer, which is not built yet.
      </WarningNote>
    </div>
  );
}
