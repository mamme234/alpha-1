/**
 * Alpha's HTTP surface.
 *
 * Convex auth routes used to be mounted here. Alpha runs its own authentication
 * over its own functions instead (see `alphaAuth/`), so the only HTTP endpoint
 * left is a health check — useful for confirming that a deployment is up before
 * pointing a client at it.
 *
 * Nothing here exposes data, secrets or configuration. It reports that the
 * backend is reachable and which Alpha build answered.
 */

import { httpRouter } from "convex/server";
import { httpAction } from "./_generated/server";

const http = httpRouter();

const ALPHA_API_VERSION = "step-1";

http.route({
  path: "/health",
  method: "GET",
  handler: httpAction(async () => {
    return new Response(
      JSON.stringify({
        service: "alpha-backend",
        status: "ok",
        apiVersion: ALPHA_API_VERSION,
        // Stated so a client can tell this deployment apart from any other.
        aiProviders: "none — Alpha runs its own model",
        checkedAt: new Date().toISOString(),
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
  }),
});

export default http;
