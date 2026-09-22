import './globals.css';

export const metadata = {
  title: 'AEGIS-IoT | AI-Driven Autonomous SOC & Threat Defense',
  description: 'Enterprise AI SOC with Live PCAP Sniffing, XGBoost Detection, TreeSHAP Explainability, Adaptive Hybrid RAG, LangGraph Multi-Agent Orchestration, and HITL Mitigation.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
