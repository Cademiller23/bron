export const ATTACHED_DATASET_STORAGE_KEY = "f1-agent-attached-dataset";

export type AttachedDatasetMetadata = {
  lastModified: number;
  name: string;
  size: number;
  type: string;
};

export type AttachedDatasetWindow = Window & {
  __f1AgentAttachedDataset?: File;
};

export function createAttachedDatasetMetadata(file: File): AttachedDatasetMetadata {
  return {
    lastModified: file.lastModified,
    name: file.name,
    size: file.size,
    type: file.type || "unknown",
  };
}

export function storeAttachedDataset(file: File) {
  const metadata = createAttachedDatasetMetadata(file);
  const datasetWindow = window as AttachedDatasetWindow;

  datasetWindow.__f1AgentAttachedDataset = file;
  window.sessionStorage.setItem(ATTACHED_DATASET_STORAGE_KEY, JSON.stringify(metadata));

  return metadata;
}

export function getAttachedDataset() {
  const datasetWindow = window as AttachedDatasetWindow;
  const file = datasetWindow.__f1AgentAttachedDataset;
  const storedMetadata = window.sessionStorage.getItem(ATTACHED_DATASET_STORAGE_KEY);

  if (file) {
    return {
      file,
      metadata: createAttachedDatasetMetadata(file),
    };
  }

  if (!storedMetadata) {
    return null;
  }

  try {
    return {
      file: undefined,
      metadata: JSON.parse(storedMetadata) as AttachedDatasetMetadata,
    };
  } catch {
    return null;
  }
}

export function formatFileSize(size: number) {
  if (size < 1024) {
    return `${size} B`;
  }

  if (size < 1024 * 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }

  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}
