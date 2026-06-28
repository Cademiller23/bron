export const CONVERSATION_SUMMARY_STORAGE_KEY =
  "f1-agent-conversation-summary";

export const CONVERSATION_SUMMARY_UPDATED_EVENT =
  "f1-agent-conversation-summary:updated";

export function readConversationSummary() {
  if (typeof window === "undefined") {
    return "";
  }

  return window.sessionStorage.getItem(CONVERSATION_SUMMARY_STORAGE_KEY) ?? "";
}

export function publishConversationSummary(summary: string) {
  window.sessionStorage.setItem(CONVERSATION_SUMMARY_STORAGE_KEY, summary);
  window.dispatchEvent(
    new CustomEvent(CONVERSATION_SUMMARY_UPDATED_EVENT, {
      detail: summary,
    }),
  );
}
