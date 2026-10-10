// Shared types and constants used by the vector-space app.
export const COLORS = {
  dataset: "#7aa2ff",
  datasetEdge: "#0f1115",
  newPoint: "#ff6b9b",
  axis: "#3a4257",
  grid: "#252b3a",
};

export type Dim = 2 | 3;

export type DatasetItem = { id: string; label: string };
export type ProjectedItem = { id: string; label: string; x: number; y: number; z: number };
export type NewPoint = ProjectedItem;
export type ExtractorInfo = { extractor: unknown; modelId: string };

// Built-in sample so users can see the tool work without typing anything.
export const EXAMPLE_DATASET = [
  "a photo of a red rose",
  "a bouquet of tulips",
  "a sunflower in a field",
  "a bowl of fresh pasta",
  "a slice of pepperoni pizza",
  "a cup of black coffee",
  "a glass of orange juice",
  "a thunderstorm in the mountains",
  "rain pattering on a window",
  "ocean waves crashing on rocks",
  "a violin playing a soft melody",
  "a piano concerto in a concert hall",
  "a guitar solo around a campfire",
  "a tabby cat sleeping on a couch",
  "a golden retriever playing fetch",
  "a hummingbird hovering at a flower",
  "a basketball game in a stadium",
  "a tennis match on a clay court",
  "a chess match in a quiet park",
  "a marathon runner crossing the finish line",
  "a programmer debugging a tricky bug",
  "an artist painting a sunset",
  "a chef plating a gourmet dish",
  "a scientist peering into a microscope",
  "a teacher writing on a chalkboard",
  "a child building a sandcastle",
  "a toddler taking their first steps",
  "a teenager riding a skateboard",
  "an elderly couple walking hand in hand",
  "a baby laughing at a peek-a-boo",
];