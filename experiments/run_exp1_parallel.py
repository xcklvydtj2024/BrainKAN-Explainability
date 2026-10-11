import subprocess
import sys
import time

def run_chunks(start_chunk, end_chunk, max_workers=4):
    processes = []
    chunk = start_chunk
    
    while chunk < end_chunk or processes:
        # Check and remove finished processes
        for p in processes[:]:
            if p.poll() is not None:
                processes.remove(p)
                print(f"[Done] A chunk finished. Running processes: {len(processes)}")
                
        # Start new processes if we have capacity and more chunks to run
        while len(processes) < max_workers and chunk < end_chunk:
            print(f"-> Starting Chunk {chunk}")
            p = subprocess.Popen([
                sys.executable, "-u", "experiments/exp1_high_precision_permutation.py",
                "--chunk_id", str(chunk),
                "--chunk_size", "100"
            ])
            processes.append(p)
            chunk += 1
            
        time.sleep(5)

if __name__ == "__main__":
    # We already ran chunk 0 and chunk 1 is running, so we start from chunk 2 to 4 (to reach 500 total)
    run_chunks(2, 5, max_workers=3)
