/**
 * Alpha Authentication — the client boundary.
 *
 * Alpha owns its own accounts and sessions. Nothing here talks to an identity
 * provider: registration, sign-in, sign-out, session expiry and revocation are
 * all Alpha functions in `src/convex/alphaAuth`.
 *
 * The client holds one piece of state — an opaque session token — and sends it
 * with each call. The backend resolves it to an account and rejects anything
 * stale, expired or revoked, so the client can never assert who it is.
 *
 * The provider is wrapped around the app once (see `src/main.tsx`); `useAuth()`
 * is how a route or component asks whether there is an account and acts on it.
 */

import { api } from "@/convex/_generated/api";
import { useAction, useMutation, useQuery } from "convex/react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

/* -------------------------------------------------------------------------- */
/* Session storage                                                             */
/* -------------------------------------------------------------------------- */

const STORAGE_KEY = "alpha.session.token";

/**
 * The token is the only secret the client holds, and the backend stores only
 * `sha256(token)`, so this copy is the single one. It lives in `localStorage`,
 * which any script on this origin can read; the server-side controls are what
 * make that survivable — absolute expiry, idle expiry, and revocation.
 */
function readStoredToken(): string | null {
  try {
    if (typeof window === "undefined" || !window.localStorage) return null;
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value && value.length > 0 ? value : null;
  } catch {
    return null;
  }
}

function storeToken(token: string): void {
  try {
    window.localStorage?.setItem(STORAGE_KEY, token);
  } catch {
    // A blocked write only means this tab will not stay signed in.
  }
}

function forgetToken(): void {
  try {
    window.localStorage?.removeItem(STORAGE_KEY);
  } catch {
    // The token is already unusable server-side once it is revoked.
  }
}

/* -------------------------------------------------------------------------- */
/* Types                                                                       */
/* -------------------------------------------------------------------------- */

export type AlphaAuthUser = {
  id: string;
  email: string;
  displayName: string;
  role: "user" | "admin";
  status: "active" | "suspended";
  createdAt: number;
  lastSignInAt: number | null;
};

export type AlphaAuthSession = {
  createdAt: number;
  expiresAt: number;
  lastSeenAt: number;
  userAgent: string | null;
};

export type AlphaSessionSummary = AlphaAuthSession & {
  id: string;
  current: boolean;
};

export type AlphaCredentials = {
  email: string;
  password: string;
  displayName?: string;
};

export type AlphaAuthContextValue = {
  /** loading while a stored token is being checked, never for a signed-out visitor. */
  status: "loading" | "signed-in" | "signed-out";
  isLoading: boolean;
  isAuthenticated: boolean;
  user: AlphaAuthUser | null;
  session: AlphaAuthSession | null;
  /** Sessions on this account, for the account controls. */
  sessions: AlphaSessionSummary[];
  sessionToken: string | null;
  signIn: (credentials: AlphaCredentials) => Promise<AlphaAuthUser>;
  signUp: (credentials: AlphaCredentials) => Promise<AlphaAuthUser>;
  signOut: () => Promise<void>;
  signOutEverywhere: () => Promise<void>;
  revokeSession: (sessionId: string) => Promise<void>;
  changePassword: (input: { currentPassword: string; newPassword: string }) => Promise<void>;
  error: string | null;
  clearError: () => void;
};

/* -------------------------------------------------------------------------- */
/* Errors                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * Alpha's backend rejects with a structured code and a message written for a
 * person. Prefer the server's wording; fall back to the transport's.
 */
export function alphaAuthErrorMessage(error: unknown): string {
  const data = (error as { data?: unknown } | null)?.data;
  if (data && typeof data === "object" && "message" in data) {
    const message = (data as { message?: unknown }).message;
    if (typeof message === "string" && message.length > 0) return message;
  }
  if (error instanceof Error && error.message) {
    // Convex prefixes uncaught server errors; the tail is the useful part.
    return error.message.replace(/^\[CONVEX[^\]]*\]\s*/i, "").replace(/^Server Error\s*/i, "") || "Something went wrong.";
  }
  return "Something went wrong. Please try again.";
}

/* -------------------------------------------------------------------------- */
/* Provider                                                                    */
/* -------------------------------------------------------------------------- */

const AlphaAuthContext = createContext<AlphaAuthContextValue | null>(null);

const userAgent = () => (typeof navigator === "undefined" ? undefined : navigator.userAgent);

export function AlphaAuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => readStoredToken());
  const [error, setError] = useState<string | null>(null);

  const signInAction = useAction(api.alphaAuth.actions.signIn);
  const registerAction = useAction(api.alphaAuth.actions.register);
  const changePasswordAction = useAction(api.alphaAuth.actions.changePassword);
  const signOutMutation = useMutation(api.alphaAuth.sessions.signOut);
  const signOutEverywhereMutation = useMutation(api.alphaAuth.sessions.signOutEverywhere);
  const revokeSessionMutation = useMutation(api.alphaAuth.sessions.revokeSession);
  const touchMutation = useMutation(api.alphaAuth.sessions.touch);

  // Reactive: if the session is revoked anywhere, this query re-runs and the
  // client is signed out without needing to poll or guess.
  const identity = useQuery(api.alphaAuth.sessions.current, token ? { token } : "skip");
  const sessions = useQuery(api.alphaAuth.sessions.listMine, token ? { token } : "skip") ?? [];

  const tokenRef = useRef<string | null>(token);
  tokenRef.current = token;

  // A token the backend no longer accepts is dropped immediately.
  useEffect(() => {
    if (token && identity === null) {
      forgetToken();
      setToken(null);
    }
  }, [token, identity]);

  // Keep the idle window honest without a write on every call.
  useEffect(() => {
    if (!token) return undefined;
    const timer = window.setInterval(() => {
      void touchMutation({ token }).catch(() => undefined);
    }, 5 * 60 * 1000);
    return () => window.clearInterval(timer);
  }, [token, touchMutation]);

  const adopt = useCallback((result: { token: string; user: AlphaAuthUser }) => {
    storeToken(result.token);
    setToken(result.token);
    setError(null);
    return result.user;
  }, []);

  const signIn = useCallback(
    async (credentials: AlphaCredentials) => {
      setError(null);
      try {
        const result = await signInAction({
          email: credentials.email,
          password: credentials.password,
          userAgent: userAgent(),
        });
        return adopt({ token: result.token, user: result.user });
      } catch (caught) {
        const message = alphaAuthErrorMessage(caught);
        setError(message);
        throw new Error(message);
      }
    },
    [adopt, signInAction],
  );

  const signUp = useCallback(
    async (credentials: AlphaCredentials) => {
      setError(null);
      try {
        const result = await registerAction({
          email: credentials.email,
          password: credentials.password,
          displayName: credentials.displayName,
          userAgent: userAgent(),
        });
        return adopt({ token: result.token, user: result.user });
      } catch (caught) {
        const message = alphaAuthErrorMessage(caught);
        setError(message);
        throw new Error(message);
      }
    },
    [adopt, registerAction],
  );

  const signOut = useCallback(async () => {
    const current = tokenRef.current;
    // Clear locally first: signing out must work even if the network call fails.
    forgetToken();
    setToken(null);
    setError(null);
    if (!current) return;
    try {
      await signOutMutation({ token: current });
    } catch {
      // The session will expire on its own; the client is already signed out.
    }
  }, [signOutMutation]);

  const signOutEverywhere = useCallback(async () => {
    const current = tokenRef.current;
    forgetToken();
    setToken(null);
    if (!current) return;
    await signOutEverywhereMutation({ token: current });
  }, [signOutEverywhereMutation]);

  const revokeSession = useCallback(
    async (sessionId: string) => {
      const current = tokenRef.current;
      if (!current) return;
      await revokeSessionMutation({ token: current, sessionId });
      if (sessionId === sessions.find((entry) => entry.current)?.id) {
        forgetToken();
        setToken(null);
      }
    },
    [revokeSessionMutation, sessions],
  );

  const changePassword = useCallback(
    async (input: { currentPassword: string; newPassword: string }) => {
      const current = tokenRef.current;
      if (!current) throw new Error("Sign in again before changing your password.");
      // Changing the password retires every session, including this one, so the
      // backend hands back a fresh token for the device that made the change.
      const result = await changePasswordAction({
        token: current,
        currentPassword: input.currentPassword,
        newPassword: input.newPassword,
        userAgent: userAgent(),
      });
      storeToken(result.token);
      setToken(result.token);
    },
    [changePasswordAction],
  );

  const value = useMemo<AlphaAuthContextValue>(() => {
    const checking = token !== null && identity === undefined;
    const user = (identity?.user as AlphaAuthUser | undefined) ?? null;
    return {
      status: checking ? "loading" : user ? "signed-in" : "signed-out",
      isLoading: checking,
      isAuthenticated: Boolean(user),
      user,
      session: (identity?.session as AlphaAuthSession | undefined) ?? null,
      sessions: sessions as AlphaSessionSummary[],
      sessionToken: user ? token : null,
      signIn,
      signUp,
      signOut,
      signOutEverywhere,
      revokeSession,
      changePassword,
      error,
      clearError: () => setError(null),
    };
  }, [changePassword, error, identity, revokeSession, sessions, signIn, signOut, signOutEverywhere, token]);

  return <AlphaAuthContext.Provider value={value}>{children}</AlphaAuthContext.Provider>;
}

export function useAuth(): AlphaAuthContextValue {
  const context = useContext(AlphaAuthContext);
  if (!context) {
    throw new Error("useAuth() must be used inside AlphaAuthProvider.");
  }
  return context;
}
