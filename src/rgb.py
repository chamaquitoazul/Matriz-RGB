import os
import sys

# Suprimir salida técnica de NumPy
os.environ['NPY_DISABLE_SVML'] = '1'
os.environ['NPY_DISABLE_CPU_FEATURES'] = 'AVX512F'

# Redirigir stderr temporalmente para evitar warnings
import io
old_stderr = sys.stderr
sys.stderr = io.StringIO()

import numpy as np
import time

# Restaurar stderr
sys.stderr = old_stderr

class RGBHSVConverter:
    """
    Convertidor RGB-HSV optimizado sin branches usando NumPy vectorizado.
    
    Arquitectura:
    - Operaciones vectorizadas eliminan branches explícitos
    - NumPy usa bibliotecas optimizadas (Intel MKL, OpenBLAS) con SIMD
    - Máscaras booleanas reemplazan condicionales
    - Procesamiento masivo nativo
    """
    
    @staticmethod
    def rgb_to_hsv(rgb):
        """
        Convierte RGB a HSV sin branches usando operaciones vectorizadas.
        
        Args:
            rgb: numpy array de shape (N, 3) con valores 0-255 o (3,) para individual
            
        Returns:
            numpy array de shape (N, 3) con valores HSV [H: 0-360, S: 0-1, V: 0-1]
        """
        # Asegurar que sea array 2D para procesamiento uniforme
        rgb = np.atleast_2d(rgb).astype(np.float32)
        
        # Normalizar a [0, 1]
        rgb_norm = rgb / 255.0
        r, g, b = rgb_norm[:, 0], rgb_norm[:, 1], rgb_norm[:, 2]
        
        # Calcular min/max sin branches usando funciones vectorizadas
        max_val = np.maximum(np.maximum(r, g), b)  # V
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val
        
        # Inicializar arrays de salida
        h = np.zeros_like(max_val)
        s = np.zeros_like(max_val)
        v = max_val
        
        # Saturación sin branches
        # S = delta / max_val, pero evitar división por 0
        mask_nonzero = max_val > 1e-10
        s = np.where(mask_nonzero, delta / max_val, 0)
        
        # Hue sin branches usando máscaras booleanas
        # Solo calcular donde delta > 0
        mask_delta = delta > 1e-10
        
        # Máscaras para determinar qué componente es máximo
        max_is_r = (max_val == r) & mask_delta
        max_is_g = (max_val == g) & mask_delta  
        max_is_b = (max_val == b) & mask_delta
        
        # Calcular H usando np.where (sin branches)
        h = np.where(max_is_r, 60 * ((g - b) / delta), h)
        h = np.where(max_is_g, 60 * (2 + (b - r) / delta), h)
        h = np.where(max_is_b, 60 * (4 + (r - g) / delta), h)
        
        # Normalizar H a [0, 360) sin branches
        h = np.where(h < 0, h + 360, h)
        
        # Combinar resultados
        hsv = np.column_stack([h, s, v])
        
        # Si entrada era 1D, devolver 1D
        return hsv[0] if rgb.shape[0] == 1 else hsv
    
    @staticmethod
    def hsv_to_rgb(hsv):
        """
        Convierte HSV a RGB sin branches usando operaciones vectorizadas.
        
        Args:
            hsv: numpy array de shape (N, 3) con HSV [H: 0-360, S: 0-1, V: 0-1]
            
        Returns:
            numpy array de shape (N, 3) con valores RGB 0-255
        """
        # Asegurar que sea array 2D
        hsv = np.atleast_2d(hsv).astype(np.float32)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        
        # Calcular componentes intermedios
        c = v * s  # Chroma
        h_norm = h / 60.0
        x = c * (1 - np.abs((h_norm % 2) - 1))
        m = v - c
        
        # Inicializar componentes RGB
        r_prime = np.zeros_like(h)
        g_prime = np.zeros_like(h)
        b_prime = np.zeros_like(h)
        
        # Determinar sector sin branches usando máscaras
        # Sector 0: [0, 60)
        mask0 = (h_norm >= 0) & (h_norm < 1)
        r_prime = np.where(mask0, c, r_prime)
        g_prime = np.where(mask0, x, g_prime)
        b_prime = np.where(mask0, 0, b_prime)
        
        # Sector 1: [60, 120)
        mask1 = (h_norm >= 1) & (h_norm < 2)
        r_prime = np.where(mask1, x, r_prime)
        g_prime = np.where(mask1, c, g_prime)
        b_prime = np.where(mask1, 0, b_prime)
        
        # Sector 2: [120, 180)
        mask2 = (h_norm >= 2) & (h_norm < 3)
        r_prime = np.where(mask2, 0, r_prime)
        g_prime = np.where(mask2, c, g_prime)
        b_prime = np.where(mask2, x, b_prime)
        
        # Sector 3: [180, 240)
        mask3 = (h_norm >= 3) & (h_norm < 4)
        r_prime = np.where(mask3, 0, r_prime)
        g_prime = np.where(mask3, x, g_prime)
        b_prime = np.where(mask3, c, b_prime)
        
        # Sector 4: [240, 300)
        mask4 = (h_norm >= 4) & (h_norm < 5)
        r_prime = np.where(mask4, x, r_prime)
        g_prime = np.where(mask4, 0, g_prime)
        b_prime = np.where(mask4, c, b_prime)
        
        # Sector 5: [300, 360)
        mask5 = (h_norm >= 5) & (h_norm < 6)
        r_prime = np.where(mask5, c, r_prime)
        g_prime = np.where(mask5, 0, g_prime)
        b_prime = np.where(mask5, x, b_prime)
        
        # Convertir a RGB final
        r = (r_prime + m) * 255
        g = (g_prime + m) * 255
        b = (b_prime + m) * 255
        
        # Redondear y asegurar rango [0, 255]
        rgb = np.column_stack([r, g, b])
        rgb = np.round(rgb).astype(np.uint8)
        
        # Si entrada era 1D, devolver 1D
        return rgb[0] if hsv.shape[0] == 1 else rgb

def cli_app():
    """Aplicación CLI simple para conversiones RGB-HSV"""
    
    while True:
        print("1. RGB → HSV")
        print("2. HSV → RGB") 
        print("3. Múltiples RGB → HSV")
        print("4. Múltiples HSV → RGB")
        print("0. Salir")
        
        try:
            choice = int(input("Opción: "))
        except ValueError:
            print("Opción inválida")
            continue
        
        if choice == 0:
            print("¡Gracias por usar RGB-HSV Converter!")
            break
        elif choice == 1:
            # RGB → HSV individual
            print("\n=== Conversión Individual RGB → HSV ===")
            try:
                r = int(input("R (0-255): "))
                g = int(input("G (0-255): "))
                b = int(input("B (0-255): "))
                
                if not all(0 <= x <= 255 for x in [r, g, b]):
                    print(" Valores deben estar entre 0-255\n")
                    continue
                
                rgb = np.array([r, g, b])
                hsv = RGBHSVConverter.rgb_to_hsv(rgb)
                
                print(f"\n Resultado:")
                print(f"RGB({r}, {g}, {b}) → HSV({hsv[0]:.0f}°, {hsv[1]*100:.1f}%, {hsv[2]*100:.1f}%)")
                
                # Verificación
                rgb_back = RGBHSVConverter.hsv_to_rgb(hsv)
                print(f" Verificación: RGB({rgb_back[0]}, {rgb_back[1]}, {rgb_back[2]})\n")
                
            except ValueError:
                print(" Valores inválidos\n")
        
        elif choice == 2:
            # HSV → RGB individual
            print("\nHSV → RGB")
            try:
                h = float(input("H (0-360°): "))
                s = float(input("S (0-100%): "))
                v = float(input("V (0-100%): "))
                
                if not (0 <= h <= 360 and 0 <= s <= 100 and 0 <= v <= 100):
                    print("H: 0-360°, S y V: 0-100%")
                    continue
                
                # Convertir porcentajes a decimales internamente
                hsv = np.array([h, s/100.0, v/100.0])
                rgb = RGBHSVConverter.hsv_to_rgb(hsv)
                
                print(f"HSV({h:.0f}°, {s:.0f}%, {v:.0f}%) → RGB({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})")
                
                # Verificación
                hsv_back = RGBHSVConverter.rgb_to_hsv(rgb)
                print(f"Verificación: HSV({hsv_back[0]:.0f}°, {hsv_back[1]*100:.0f}%, {hsv_back[2]*100:.0f}%)")
                
            except ValueError:
                print("Valores inválidos")
        
        elif choice == 3:
            # Múltiples RGB → HSV
            print("\n=== Conversión Múltiple RGB → HSV ===")
            try:
                count = int(input("Número de colores a generar (1-10000000): "))
                if not 1 <= count <= 10000000:
                    print(" Número debe estar entre 1 y 10,000,000\n")
                    continue
                
                print(f" Generando {count:,} colores aleatorios...")
                
                # Generar datos aleatorios eficientemente
                rgb_data = np.random.randint(0, 256, size=(count, 3), dtype=np.uint8)
                
                print(f" Procesando {count:,} colores con NumPy vectorizado...")
                
                start_time = time.time()
                hsv_data = RGBHSVConverter.rgb_to_hsv(rgb_data)
                end_time = time.time()
                
                duration = (end_time - start_time) * 1000  # ms
                speed = count / (duration / 1000) if duration > 0 else float('inf')  # conversiones/segundo
                
                print(f"✅ Conversión completada en {duration:.2f} ms")
                if duration > 0:
                    print(f" Velocidad: {speed:,.0f} conversiones/segundo")
                else:
                    print(f" Velocidad: >1,000,000 conversiones/segundo (muy rápido para medir)")
                
                # Mostrar primeros resultados
                print(f"\n Primeros 5 resultados:")
                for i in range(min(5, count)):
                    r, g, b = rgb_data[i]
                    h, s, v = hsv_data[i]
                    print(f"RGB({r},{g},{b}) → HSV({h:.0f}°,{s*100:.1f}%,{v*100:.1f}%)")
                
                if count > 5:
                    print(f"... (y {count-5:,} más)\n")
                
            except ValueError:
                print(" Valor inválido\n")
        
        elif choice == 4:
            # Múltiples HSV → RGB
            print("\n=== Conversión Múltiple HSV → RGB ===")
            try:
                count = int(input("Número de colores a generar (1-10000000): "))
                if not 1 <= count <= 10000000:
                    print("❌ Número debe estar entre 1 y 10,000,000\n")
                    continue
                
                print(f" Generando {count:,} colores HSV aleatorios...")
                
                # Generar datos HSV aleatorios
                h = np.random.uniform(0, 360, count)
                s = np.random.uniform(0, 1, count)
                v = np.random.uniform(0, 1, count)
                hsv_data = np.column_stack([h, s, v])
                
                print(f" Procesando {count:,} colores con NumPy vectorizado...")
                
                start_time = time.time()
                rgb_data = RGBHSVConverter.hsv_to_rgb(hsv_data)
                end_time = time.time()
                
                duration = (end_time - start_time) * 1000  # ms
                speed = count / duration * 1000  # conversiones/segundo
                
                print(f"Conversión completada en {duration:.2f} ms")
                print(f" Velocidad: {speed:,.0f} conversiones/segundo")
                
                # Mostrar primeros resultados
                print(f"\n Primeros 5 resultados:")
                for i in range(min(5, count)):
                    h_val, s_val, v_val = hsv_data[i]
                    r, g, b = rgb_data[i]
                    print(f"HSV({h_val:.0f}°,{s_val*100:.1f}%,{v_val*100:.1f}%) → RGB({int(r)},{int(g)},{int(b)})")
                
                if count > 5:
                    print(f"... (y {count-5:,} más)\n")
                
            except ValueError:
                print(" Valor inválido\n")
        
        else:
            print(" Opción inválida\n")

if __name__ == "__main__":
    cli_app()

