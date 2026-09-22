import subprocess
import sys
import os
import time
import signal

def main():
    print("====================================================================")
    print("           AEGIS-IoT: AI-DRIVEN AUTONOMOUS SOC PLATFORM             ")
    print("====================================================================")
    print("• Detection Engine : XGBoost (< 0.14 ms) + TreeSHAP Explainability")
    print("• RAG Architecture : Dense + BM25 + Reciprocal Rank Fusion (RRF)")
    print("• Orchestrator     : LangGraph Supervisor Multi-Agent System")
    print("• SOC Dashboard    : Next.js Real-Time Cyber Command Center")
    print("• Ingestion Layer  : Live PCAP Sniffing + 5-Tuple Real-Time Extractor")
    print("====================================================================")
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(root_dir, "frontend")

    # 1. Start FastAPI Backend Server
    print("\n[1/2] Launching AEGIS-IoT Backend Server on http://127.0.0.1:8000 ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api.server:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=root_dir
    )

    time.sleep(2)

    # 2. Start Next.js Frontend Dev Server
    print("\n[2/2] Launching AEGIS-IoT Next.js SOC Dashboard on http://localhost:3000 ...")
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=frontend_dir
    )

    print("\n====================================================================")
    print(">> AEGIS-IoT Platform is now LIVE!")
    print(">> Open SOC Dashboard : http://localhost:3000")
    print(">> Backend API & Docs : http://127.0.0.1:8000/docs")
    print("====================================================================")
    print("Press Ctrl+C to stop all services.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping AEGIS-IoT servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
