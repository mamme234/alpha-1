/**
 * Alpha Authentication page.
 *
 * Registration and sign-in against Alpha's own accounts. There is no identity
 * provider behind this form, no one-time code emailed by a third party, and no
 * guest identity that would let a visitor act as an account.
 *
 * The form states the password rules the server enforces rather than revealing
 * them one rejection at a time, and sign-in failures stay deliberately vague:
 * "email or password is incorrect" is the same answer whether the address exists
 * or the password was wrong, so this page cannot be used to discover who has an
 * account.
 */

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Eyebrow, Mono, Rule } from "@/components/alpha/studio";
import { alphaAuthErrorMessage, useAuth } from "@/hooks/use-auth";
import { ArrowRight, Loader2, LockKeyhole, ShieldCheck, TriangleAlert } from "lucide-react";
import { Suspense, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router";

interface AuthProps {
  redirectAfterAuth?: string;
}

const PASSWORD_MIN_LENGTH = 10;

function resolveRedirectAfterAuth(returnTo: string | null, fallback = "/dashboard") {
  if (returnTo?.startsWith("/") && !returnTo.startsWith("//")) {
    return returnTo;
  }
  return fallback;
}

function Auth({ redirectAfterAuth }: AuthProps = {}) {
  const { isLoading: authLoading, isAuthenticated, signIn, signUp } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const redirect = resolveRedirectAfterAuth(searchParams.get("returnTo"), redirectAfterAuth);

  const [mode, setMode] = useState<"signIn" | "signUp">("signIn");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      navigate(redirect, { replace: true });
    }
  }, [authLoading, isAuthenticated, navigate, redirect]);

  const switchMode = (next: "signIn" | "signUp") => {
    setMode(next);
    setError(null);
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsSubmitting(true);
    setError(null);
    try {
      if (mode === "signUp") {
        await signUp({ email, password, displayName: displayName.trim() || undefined });
      } else {
        await signIn({ email, password });
      }
      navigate(redirect, { replace: true });
    } catch (caught) {
      setError(alphaAuthErrorMessage(caught));
    } finally {
      setIsSubmitting(false);
    }
  };

  const longEnough = password.length >= PASSWORD_MIN_LENGTH;
  const avoidsEmail = email.length === 0 || !password.toLowerCase().includes(email.trim().toLowerCase());

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex max-w-5xl flex-col px-6 py-10">
        <button
          type="button"
          onClick={() => navigate("/")}
          className="studio-serif w-fit text-lg tracking-tight"
        >
          Alpha
        </button>
        <Eyebrow className="mt-1">self-owned AI system</Eyebrow>

        <div className="mt-10 grid gap-10 lg:grid-cols-[1fr_minmax(340px,380px)]">
          {/* Editorial side: says what this account is for, honestly. */}
          <section className="space-y-6">
            <h1 className="studio-serif text-3xl leading-tight">
              {mode === "signUp" ? "Create your Alpha account" : "Sign in to Alpha"}
            </h1>
            <p className="max-w-md text-sm leading-6 text-muted-foreground">
              An Alpha account keeps your training runs, checkpoints, documents, vectors and memories tied to you
              alone. Alpha resolves every request from the session you create here — the client never names its own
              identity.
            </p>
            <Rule />
            <ul className="space-y-3 text-xs leading-5 text-muted-foreground">
              <li className="flex gap-3">
                <ShieldCheck className="mt-0.5 size-3.5 shrink-0 text-chart-2" />
                <span>
                  Passwords are stored as a PBKDF2-HMAC-SHA256 derivation with a per-account salt. The password itself
                  is never written anywhere.
                </span>
              </li>
              <li className="flex gap-3">
                <LockKeyhole className="mt-0.5 size-3.5 shrink-0 text-chart-2" />
                <span>
                  Sessions expire after 30 days, stop working after 7 days of inactivity, and can be revoked from the
                  workspace. Changing your password ends all of them.
                </span>
              </li>
              <li className="flex gap-3">
                <ShieldCheck className="mt-0.5 size-3.5 shrink-0 text-chart-2" />
                <span>
                  No external AI provider is involved. Alpha's own model is the engine, and its state —{" "}
                  <Mono>UNTRAINED</Mono> until a run completes — is shown rather than implied.
                </span>
              </li>
            </ul>
          </section>

          {/* Form side */}
          <section className="studio-frame h-fit">
            <div className="flex items-center gap-1 border-b border-border px-4 py-3">
              {(["signIn", "signUp"] as const).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => switchMode(option)}
                  className={`rounded-sm px-3 py-1 text-[11px] uppercase tracking-[0.14em] transition-colors ${
                    mode === option
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {option === "signIn" ? "Sign in" : "Create account"}
                </button>
              ))}
            </div>

            <form onSubmit={handleSubmit} className="space-y-4 px-4 py-5">
              <div className="space-y-1.5">
                <Label htmlFor="email" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                  Email
                </Label>
                <Input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  value={email}
                  onChange={(event) => setEmail(event.target.value)}
                  placeholder="you@example.com"
                  disabled={isSubmitting}
                />
              </div>

              {mode === "signUp" ? (
                <div className="space-y-1.5">
                  <Label htmlFor="displayName" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                    Display name <span className="normal-case tracking-normal">(optional)</span>
                  </Label>
                  <Input
                    id="displayName"
                    name="displayName"
                    autoComplete="nickname"
                    value={displayName}
                    onChange={(event) => setDisplayName(event.target.value)}
                    placeholder="How Alpha addresses you"
                    disabled={isSubmitting}
                  />
                </div>
              ) : null}

              <div className="space-y-1.5">
                <Label htmlFor="password" className="text-[11px] uppercase tracking-[0.14em] text-muted-foreground">
                  Password
                </Label>
                <Input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete={mode === "signUp" ? "new-password" : "current-password"}
                  required
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  disabled={isSubmitting}
                />
                {mode === "signUp" ? (
                  <ul className="space-y-1 pt-1 text-[11px] leading-4 text-muted-foreground">
                    <li className={longEnough ? "text-chart-2" : undefined}>
                      {longEnough ? "✓" : "•"} at least {PASSWORD_MIN_LENGTH} characters
                    </li>
                    <li className={avoidsEmail ? "text-chart-2" : undefined}>
                      {avoidsEmail ? "✓" : "•"} does not contain your email address
                    </li>
                    <li>• not a password from published breach lists</li>
                  </ul>
                ) : null}
              </div>

              {error ? (
                <p className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-[11px] leading-4 text-foreground">
                  <TriangleAlert className="mt-0.5 size-3.5 shrink-0 text-destructive" />
                  <span>{error}</span>
                </p>
              ) : null}

              <Button type="submit" className="w-full" disabled={isSubmitting || !email || !password}>
                {isSubmitting ? (
                  <Loader2 className="mr-2 size-3.5 animate-spin" />
                ) : (
                  <ArrowRight className="mr-2 size-3.5" />
                )}
                {mode === "signUp" ? "Create account" : "Sign in"}
              </Button>

              <p className="text-[11px] leading-4 text-muted-foreground">
                {mode === "signUp"
                  ? "Already have an account? Switch to sign in."
                  : "New here? Switch to create account — no email verification step is involved because Alpha sends no mail."}
              </p>
            </form>

            <div className="border-t border-border px-4 py-3">
              <p className="text-[11px] leading-4 text-muted-foreground">
                Secured by <span className="text-foreground">Alpha</span> — its own accounts, its own sessions, its own
                audit log.
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}

export default function AuthPage(props: AuthProps) {
  return (
    <Suspense>
      <Auth {...props} />
    </Suspense>
  );
}
