import { eveChannel } from "eve/channels/eve";
import { localDev, none, vercelOidc, type AuthFn } from "eve/channels/auth";

// Optional shared-secret gate for public demos: when EVE_PUBLIC_BEARER_TOKEN is
// set, only requests carrying `Authorization: Bearer <token>` authenticate.
// When unset, the channel stays open (public demo) via the anonymous fallback.
const bearerTokenAuth: AuthFn<Request> = async (request) => {
  const expectedToken = process.env.EVE_PUBLIC_BEARER_TOKEN?.trim();
  if (!expectedToken) return null;
  const header = request.headers.get("authorization") ?? "";
  if (header !== `Bearer ${expectedToken}`) return null;
  return {
    attributes: {},
    authenticator: "bearer",
    principalId: "journalist-authorized",
    principalType: "user",
  };
};

export default eveChannel({
  auth: [
    // Lets the eve TUI and other Vercel deployments in this project reach the agent.
    vercelOidc(),
    // Open on localhost for `eve dev`; ignored in production deployments.
    localDev(),
    bearerTokenAuth,
    // Public demo fallback: accepts any request anonymously. To require auth
    // in production, set EVE_PUBLIC_BEARER_TOKEN and remove none().
    none(),
  ],
});
