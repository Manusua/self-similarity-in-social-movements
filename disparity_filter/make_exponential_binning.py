import math

def main():
    N = 60
    LARGO = N * N
    TIEMPO_TOMA = 1000 * LARGO * 7   # 1,000 × LARGO × 7
    aa = 1.02

    s = [0] * (LARGO + 1)
    frecuency = [0.0] * (2 * LARGO + 1)

    normalizc = float(LARGO) * float(TIEMPO_TOMA)

    # Read input_binning.dat (two columns: int, float)
    with open("input_binning.dat", "r") as fpentrada:
        for i in range(1, LARGO + 1):
            line = fpentrada.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) >= 2:
                s[i] = int(parts[0])
                frecuency[i] = float(parts[1])

    # Scale frequencies exactly as in the C code
    for i in range(1, LARGO + 1):
        frecuency[i] *= normalizc

    with open("output_binning.dat", "w") as fpsalida:
        # Re‑initialize i for the binning loop
        i = 1
        while aa**i <= LARGO:
            desde = aa**i
            hasta = aa**(i + 1)
            # If exactly integer, nudge by 0.1 to avoid edge‐cases
            if math.floor(desde) == desde:
                desde -= 0.1
            if math.floor(hasta) == hasta:
                hasta += 0.1

            # figure out how many bins (ancho)
            _, primer = math.modf(desde)
            _, segundo = math.modf(hasta)
            ancho = int(segundo) - int(primer)

            # Only if there is at least one bin, do the detailed work
            if ancho > 0:
                #shift the lower edge by +1 to define the actual bin
                desdemas1 = desde + 1
                _, primer = math.modf(desdemas1)
                _, segundo = math.modf(hasta)
                if segundo >= LARGO:
                    segundo = LARGO
                    ancho = int(segundo) - int(primer)

                # Geometric‐mean  
                geomean = math.sqrt(primer * segundo)
                total = 0.0
                for j in range(int(primer), int(segundo) + 1):
                    total += frecuency[j]

                # Normalize exactly as in C
                valorc = total / (normalizc * ancho)
                fpsalida.write(f"{geomean:17.10e} {valorc:17.10e}\n")
            i += 1

if __name__ == "__main__":
    main()
