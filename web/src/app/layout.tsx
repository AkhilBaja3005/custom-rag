import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SOTA RAG Studio - 10,000 Page Enterprise Engine",
  description: "Enterprise State-of-the-Art RAG Platform with Live SSE Streaming & Graph RAG",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="bg-zinc-950 text-zinc-100 antialiased selection:bg-purple-500 selection:text-white min-h-screen">
        {children}
      </body>
    </html>
  );
}
