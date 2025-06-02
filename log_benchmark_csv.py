import csv

def save_benchmark_csv(custom_times, scipy_times, sizes, filename="benchmark_results.csv"):
    """
    Salva i risultati del benchmark base in formato CSV con notazione scientifica
    
    Args:
        custom_times: lista dei tempi dell'implementazione custom
        scipy_times: lista dei tempi dell'implementazione SciPy
        sizes: lista delle dimensioni delle matrici testate
        filename: nome del file CSV di output
    
    Returns:
        str: percorso del file salvato
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Header del CSV
        writer.writerow(['Matrix_Size', 'Custom_Time_s', 'SciPy_Time_s', 'Speedup'])
        
        # Scrivi i dati con notazione scientifica
        for i, size in enumerate(sizes):
            print(f"Processing size: {size}x{size}")
            row = [
                size,
                f"{custom_times[i]:.2e}",
                f"{scipy_times[i]:.2e}",
            ]
            writer.writerow(row)
    
    print(f"Risultati salvati in CSV: {filename}")
    return filename