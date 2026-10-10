import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vector Space Explorer",
  description: "Embed your data, project to 2D or 3D with PCA, and drop in new points to see where they land.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}