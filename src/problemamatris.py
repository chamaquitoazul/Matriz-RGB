
import sys
from typing import Tuple
import time
import argparse


def get_value_optimized_pure(row: int, col: int) -> int:
    """
    Función matemática pura sin ramas ni lookup tables.
    Implementación 100% algebraica sin referencias a datos originales.
    
    Args:
        row: Fila (0-indexada)
        col: Columna (0-indexada)
    
    Returns:
        Valor predicho usando fórmulas matemáticas puras
    """
    
    # Descomposición de coordenadas usando bit manipulation
    local_col = col & 7        # col % 8
    local_row = row & 7        # row % 8
    block_row = row >> 3       # row // 8
    col_block = col >> 3       # col // 8
    
    # 1. PATRÓN BASE - Fórmula matemática algebraica para [1,0,1,2,3,2,3,0]
    # Análisis del patrón: posiciones pares vs impares siguen reglas diferentes
    
    is_even_col = 1 - (local_col & 1)  # 1 si columna par, 0 si impar
    is_odd_col = local_col & 1          # 0 si columna par, 1 si impar
    
    # Para columnas pares (0,2,4,6): valores [1,1,3,3]
    # Fórmula: 1 + 2 * (col_pos >= 4)
    even_base = 1 + 2 * (local_col >> 2)
    
    # Para columnas impares (1,3,5,7): valores [0,2,2,0]
    # Fórmula: 2 * (col_pos in [2,4]) = 2 * ((col_pos >> 1) == 1 or (col_pos >> 1) == 2)
    col_quarter = local_col >> 1  # 0,0,1,1,2,2,3,3
    odd_base = 2 * ((col_quarter == 1) | (col_quarter == 2))
    
    # Convertir comparaciones a aritmética sin ramas
    # (col_quarter == 1) → resultado 1 solo cuando quarter=1
    quarter_is_1 = (col_quarter == 1)  # 1 cuando quarter=1, 0 en otro caso
    quarter_is_2 = (col_quarter == 2)  # 1 cuando quarter=2, 0 en otro caso
    
    # Para evitar comparaciones, usar función característica:
    # f(x,target) = 1 - |sign(x - target)| donde sign da -1,0,1
    # Simplificado: (x == target) se puede expresar como 1 - min(1, |x - target|)
    
    # Método más directo usando propiedades modulares:
    odd_base = 2 * (((col_quarter - 1) == 0) | ((col_quarter - 2) == 0))
    
    # Versión sin comparaciones usando aritmética modular:
    # Crear máscara que sea 1 solo para quarter=1 o quarter=2
    mask_1 = 1 - min(1, abs(col_quarter - 1))  # 1 si quarter=1, 0 si no
    mask_2 = 1 - min(1, abs(col_quarter - 2))  # 1 si quarter=2, 0 si no
    odd_base = 2 * (mask_1 + mask_2)
    
    # Combinar valores pares e impares
    base_value = is_even_col * even_base + is_odd_col * odd_base
    
    # Corrección para columna 7 (local_col=7 debe dar 0)
    col7_correction = (local_col == 7) * base_value
    base_value = base_value - col7_correction
    
    # 2. OFFSET POR FILA - Fórmula matemática pura
    # Patrón [0,4,0,4,8,12,8,12] = 4*(row%2) + 8*(row//4)
    row_offset = ((local_row & 1) << 2) + ((local_row >> 2) << 3)
    
    # 3. ESCALAMIENTO POR BLOQUE
    block_offset = block_row << 4  # block_row * 16
    
    # 4. CASO ESPECIAL COLUMNA 15 - Fórmula matemática algebraica
    # Secuencia [4,0,4,8,12,8,12,16] - derivar fórmula sin lookup
    
    is_col_15 = (col_block == 1) & (local_col == 7)
    
    # Análisis algebraico de la secuencia [4,0,4,8,12,8,12,16]:
    # Patrón: base_value + adjustment_by_position
    
    # Base: 4 para primera mitad (rows 0-3), 12 para segunda mitad (rows 4-7)
    is_second_half = local_row >> 2  # 0 para rows 0-3, 1 para rows 4-7
    col15_base = 4 + 8 * is_second_half
    
    # Ajustes por posición específica:
    # Rows 1,5: -4 (posiciones impares en primera posición de cada mitad)
    # Rows 3,7: +4 (posiciones impares en segunda posición de cada mitad)
    
    row_in_half = local_row & 3  # Posición dentro de cada mitad de 4
    is_row_odd_in_half = row_in_half & 1  # 1 para posiciones 1,3 dentro de mitad
    is_second_pos_in_half = (row_in_half >> 1) & 1  # 1 para posiciones 2,3 dentro de mitad
    
    # Fórmula: ajuste = (-4 + 8*second_pos) * is_odd = -4*is_odd + 8*is_odd*second_pos
    col15_adjustment = is_row_odd_in_half * (-4 + 8 * is_second_pos_in_half)
    
    special_value = col15_base + col15_adjustment
    special_result = special_value + block_offset
    
    # 5. COMBINACIÓN FINAL SIN RAMAS
    normal_result = base_value + row_offset + block_offset
    
    # Selección sin condicionales usando multiplicación booleana
    final_result = normal_result * (1 - is_col_15) + special_result * is_col_15
    
    return final_result & 0xFF


def get_value_simd_optimized(row: int, col: int) -> int:
    """
    Versión optimizada para instrucciones SIMD.
    Diseñada para paralelización vectorial y alta eficiencia.
    
    Args:
        row: Fila
        col: Columna
    
    Returns:
        Valor calculado optimizado para SIMD
    """
    
    # Descomposición ultra-eficiente
    r3 = row & 7
    c3 = col & 7
    rb = row >> 3
    cb = col >> 3
    
    # Patrón base usando operaciones vectorizables
    c_even = ((c3 & 1) ^ 1)
    q = c3 >> 1
    h = c3 >> 2
    
    # Fórmula optimizada para SIMD
    base = c_even * (1 + 2 * h - 2 * (q == 3)) + (c3 & 1) * (2 * (q & 1) * (h ^ 1))
    
    # Offsets usando operaciones paralelas
    r_off = ((r3 & 1) << 2) + ((r3 & 4) << 1)
    b_off = rb << 4
    
    # Caso especial columna 15 con fórmula SIMD-friendly
    col15_mask = (cb == 1) & (c3 == 7)
    special = (4 + 8 * (r3 >> 2) - 4 * (r3 & 1) * ((r3 >> 2) ^ 1) + 
              4 * ((r3 >> 2) & ((r3 & 1) ^ 1)))
    
    # Combinación sin ramas para vectorización
    result = (base + r_off + b_off) * (col15_mask ^ 1) + (special + b_off) * col15_mask
    
    return result & 0xFF


def verify_mathematical_formula():
    """
    Verificación de las fórmulas matemáticas derivadas.
    """
    
    print("Verificando fórmulas...")
    
    # Casos de prueba basados en el dataset original
    test_cases = [
        (0, 0, 1), (0, 1, 0), (0, 2, 1), (0, 7, 0), (0, 15, 4),
        (1, 0, 5), (1, 1, 4), (1, 15, 0),
        (2, 0, 1), (2, 15, 4),
        (3, 0, 5), (3, 15, 8),
        (4, 0, 9), (4, 15, 12),
        (5, 0, 13), (5, 15, 8),
        (6, 0, 9), (6, 15, 12),
        (7, 0, 13), (7, 15, 16),
        (8, 0, 17), (8, 15, 20),
        (15, 15, 32)
    ]
    
    print("Row Col | Esperado | Obtenido | Status")
    print("-" * 40)
    
    all_correct = True
    for row, col, expected in test_cases:
        result = get_value_optimized_pure(row, col)
        status = "✅" if result == expected else "❌"
        if result != expected:
            all_correct = False
        print(f"{row:3d} {col:3d} |    {expected:3d}   |    {result:3d}   |   {status}")
    
    print(f"Resultado: {'CORRECTO' if all_correct else 'ERROR'}")
    
    # Demostración de extrapolación
    print("\nExtrapolación:")
    print("Row Col | Valor")
    print("-" * 15)
    
    extrapolation_cases = [(50, 10), (100, 15), (200, 5), (1000, 7)]
    for row, col in extrapolation_cases:
        result = get_value_optimized_pure(row, col)
        print(f"{row:3d} {col:3d} |  {result:3d}")


def performance_benchmark():
    """
    Benchmark de rendimiento de las implementaciones.
    """
    
    print("Benchmark de rendimiento...")
    
    # Test con coordenadas aleatorias
    import random
    test_coords = [(random.randint(0, 1000), random.randint(0, 100)) 
                   for _ in range(10000)]
    
    # Benchmark función optimizada
    start_time = time.time()
    for row, col in test_coords:
        get_value_optimized_pure(row, col)
    optimized_time = time.time() - start_time
    
    # Benchmark función SIMD
    start_time = time.time()
    for row, col in test_coords:
        get_value_simd_optimized(row, col)
    simd_time = time.time() - start_time
    
    print(f"Función optimizada: {optimized_time:.4f}s (10,000 cálculos)")
    print(f"Función SIMD:       {simd_time:.4f}s (10,000 cálculos)")
    print(f"Velocidad: {10000/optimized_time:.0f} cálculos/segundo")


def interactive_search():
    """
    Búsqueda interactiva de coordenadas por CLI.
    """
    
    print("Búsqueda interactiva")
    print("Formato: fila,columna o 'q' para salir")
    print()
    
    while True:
        try:
            entrada = input("Coordenadas: ").strip()
            
            if entrada.lower() in ['q', 'quit', 'exit']:
                break
            
            if ',' in entrada:
                try:
                    fila_str, columna_str = entrada.split(',', 1)
                    fila = int(fila_str.strip())
                    columna = int(columna_str.strip())
                except ValueError:
                    print("Error: Formato inválido. Usa: fila,columna")
                    continue
            else:
                try:
                    fila = int(entrada)
                    columna_str = input(f"Columna para fila {fila}: ").strip()
                    columna = int(columna_str)
                except ValueError:
                    print("Error: Números inválidos")
                    continue
            
            resultado = get_value_optimized_pure(fila, columna)
            print(f"({fila}, {columna}) = {resultado}")
            
            # Información adicional
            local_row = fila % 8
            local_col = columna % 8
            block_row = fila // 8
            is_col_15 = (columna // 8 == 1) and (local_col == 7)
            
            print(f"Detalles: local_row={local_row}, local_col={local_col}, block={block_row}")
            if is_col_15:
                print("Nota: Columna especial (15)")
            print()
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def batch_search(coordenadas_lista):
    """
    Búsqueda en lote de múltiples coordenadas.
    
    Args:
        coordenadas_lista: Lista de tuplas (fila, columna)
    """
    
    print("Fila  Col | Valor")
    print("-" * 15)
    
    for fila, columna in coordenadas_lista:
        try:
            resultado = get_value_optimized_pure(fila, columna)
            print(f"{fila:4d} {columna:4d} | {resultado:5d}")
        except Exception as e:
            print(f"{fila:4d} {columna:4d} | ERROR")


def parse_coordinates_from_args():
    """
    Parsea coordenadas desde argumentos de línea de comandos.
    """
    
    parser = argparse.ArgumentParser(
        description="Calculadora de valores del dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python problema2.py                    # Análisis completo
  python problema2.py -i                # Modo interactivo
  python problema2.py -c 5,10           # Coordenada específica
  python problema2.py -c 5,10 15,7      # Múltiples coordenadas
  python problema2.py -f coords.txt      # Desde archivo
        """
    )
    
    parser.add_argument('-i', '--interactive', action='store_true',
                      help='Modo interactivo')
    
    parser.add_argument('-c', '--coords', nargs='+', 
                      help='Coordenadas (formato: fila,columna)')
    
    parser.add_argument('-f', '--file',
                      help='Archivo con coordenadas')
    
    parser.add_argument('--no-analysis', action='store_true',
                      help='Solo buscar coordenadas')
    
    return parser.parse_args()


def read_coordinates_from_file(filename):
    """
    Lee coordenadas desde un archivo de texto.
    
    Args:
        filename: Nombre del archivo
        
    Returns:
        Lista de tuplas (fila, columna)
    """
    
    coordenadas = []
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        if ',' in line:
                            fila, columna = line.split(',', 1)
                            coordenadas.append((int(fila.strip()), int(columna.strip())))
                        else:
                            print(f"Línea {line_num} ignorada: {line}")
                    except ValueError:
                        print(f"Línea {line_num} ignorada: {line}")
    
    except FileNotFoundError:
        print(f"Error: Archivo '{filename}' no encontrado")
        return []
    except Exception as e:
        print(f"Error leyendo archivo: {e}")
        return []
    
    return coordenadas


def main():
    """
    Función principal que maneja argumentos CLI.
    """
    
    args = parse_coordinates_from_args()
    
    if args.interactive:
        interactive_search()
        return
    
    elif args.coords:
        coordenadas = []
        for coord_str in args.coords:
            try:
                if ',' in coord_str:
                    fila, columna = coord_str.split(',', 1)
                    coordenadas.append((int(fila.strip()), int(columna.strip())))
                else:
                    print(f"Coordenada ignorada: {coord_str}")
            except ValueError:
                print(f"Coordenada ignorada: {coord_str}")
        
        if coordenadas:
            if not args.no_analysis:
                print("TAGSHELF - PROBLEMA 2")
                print("Búsqueda de coordenadas\n")
            
            batch_search(coordenadas)
        return
    
    elif args.file:
        coordenadas = read_coordinates_from_file(args.file)
        if coordenadas:
            if not args.no_analysis:
                print("TAGSHELF - PROBLEMA 2")
                print(f"Leyendo desde: {args.file}\n")
            
            batch_search(coordenadas)
        return
    
    # Análisis completo por defecto
    print("TAGSHELF - PROBLEMA 2")
    print("Función matemática para predicción de valores\n")
    
    verify_mathematical_formula()
    print()
    performance_benchmark()
    
   

if __name__ == "__main__":
    main()