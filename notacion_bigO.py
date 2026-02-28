from __future__ import annotations

import os
import random
import sys
import threading
import time
import queue
from bisect import bisect_left
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# -----------------------------
# Parámetros por defecto
# -----------------------------
DEFAULT_N = 10_000_000
MINV = -50_000_000
MAXV = 50_000_000
DEFAULT_AUTO_SEARCHES = 1000
DEFAULT_TXT = "numeros_10M.txt"


# -----------------------------
# Utilidades
# -----------------------------
def human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    i = 0
    while x >= 1024.0 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"


def ns_to_us(ns: int) -> float:
    return ns / 1000.0


def approx_list_memory(n_items: int) -> int:
    # Estimación aproximada en CPython 64-bit:
    # - lista: ~8 bytes por referencia
    # - int: ~28 bytes
    base = sys.getsizeof([])
    refs = 8 * n_items
    ints = 28 * n_items
    return base + refs + ints


def approx_set_memory(n_items: int) -> int:
    # Muy aproximado (depende de load factor / implementación):
    # estimación conservadora ~72 bytes por elemento
    return int(72 * n_items)


def approx_tree_memory(n_nodes: int, per_node_est: int) -> int:
    # per_node_est: estimación por nodo (muy variable)
    return n_nodes * per_node_est


# -----------------------------
# Generar / cargar TXT
# -----------------------------
def generate_txt(path: str, n: int, seed: int | None, progress_cb) -> None:
    rng = random.Random(seed)
    chunk = 200_000

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        written = 0
        while written < n:
            k = min(chunk, n - written)
            lines = "\n".join(str(rng.randint(MINV, MAXV)) for _ in range(k))
            f.write(lines)
            f.write("\n")
            written += k
            progress_cb(written, n)


def load_txt(path: str, progress_cb) -> list[int]:
    data: list[int] = []
    app = data.append
    c = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            app(int(line))
            c += 1
            if c % 200_000 == 0:
                progress_cb(c)
    progress_cb(c)
    return data


# -----------------------------
# Arreglo ordenado + binaria
# -----------------------------
def binary_search(sorted_list: list[int], x: int) -> bool:
    i = bisect_left(sorted_list, x)
    return i != len(sorted_list) and sorted_list[i] == x


# -----------------------------
# BST
# -----------------------------
class BSTNode:
    __slots__ = ("key", "count", "left", "right")

    def __init__(self, key: int):
        self.key = key
        self.count = 1
        self.left: BSTNode | None = None
        self.right: BSTNode | None = None


class BST:
    def __init__(self):
        self.root: BSTNode | None = None
        self.nodes = 0  # nodos distintos (sin contar duplicados)

    def insert(self, key: int) -> None:
        if self.root is None:
            self.root = BSTNode(key)
            self.nodes = 1
            return

        cur = self.root
        while True:
            if key == cur.key:
                cur.count += 1
                return
            elif key < cur.key:
                if cur.left is None:
                    cur.left = BSTNode(key)
                    self.nodes += 1
                    return
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = BSTNode(key)
                    self.nodes += 1
                    return
                cur = cur.right

    def contains(self, key: int) -> bool:
        cur = self.root
        while cur is not None:
            if key == cur.key:
                return True
            cur = cur.left if key < cur.key else cur.right
        return False


# -----------------------------
# AVL
# -----------------------------
class AVLNode:
    __slots__ = ("key", "count", "left", "right", "h")

    def __init__(self, key: int):
        self.key = key
        self.count = 1
        self.left: AVLNode | None = None
        self.right: AVLNode | None = None
        self.h = 1  # altura


def _h(n: AVLNode | None) -> int:
    return n.h if n else 0


def _update(n: AVLNode) -> None:
    hl = _h(n.left)
    hr = _h(n.right)
    n.h = 1 + (hl if hl > hr else hr)


def _bf(n: AVLNode) -> int:
    return _h(n.left) - _h(n.right)


def _rot_right(y: AVLNode) -> AVLNode:
    x = y.left
    assert x is not None
    t2 = x.right

    x.right = y
    y.left = t2

    _update(y)
    _update(x)
    return x


def _rot_left(x: AVLNode) -> AVLNode:
    y = x.right
    assert y is not None
    t2 = y.left

    y.left = x
    x.right = t2

    _update(x)
    _update(y)
    return y


class AVL:
    def __init__(self):
        self.root: AVLNode | None = None
        self.nodes = 0  # nodos distintos

    def insert(self, key: int) -> None:
        self.root, added_new = self._insert(self.root, key)
        if added_new:
            self.nodes += 1

    def _insert(self, node: AVLNode | None, key: int) -> tuple[AVLNode, bool]:
        if node is None:
            return AVLNode(key), True

        added_new = False
        if key == node.key:
            node.count += 1
            return node, False
        elif key < node.key:
            node.left, added_new = self._insert(node.left, key)
        else:
            node.right, added_new = self._insert(node.right, key)

        _update(node)
        balance = _bf(node)

        # 4 casos
        # Left Left
        if balance > 1 and key < (node.left.key if node.left else key):
            return _rot_right(node), added_new
        # Right Right
        if balance < -1 and key > (node.right.key if node.right else key):
            return _rot_left(node), added_new
        # Left Right
        if balance > 1 and key > (node.left.key if node.left else key):
            node.left = _rot_left(node.left)  # type: ignore[arg-type]
            return _rot_right(node), added_new
        # Right Left
        if balance < -1 and key < (node.right.key if node.right else key):
            node.right = _rot_right(node.right)  # type: ignore[arg-type]
            return _rot_left(node), added_new

        return node, added_new

    def contains(self, key: int) -> bool:
        cur = self.root
        while cur is not None:
            if key == cur.key:
                return True
            cur = cur.left if key < cur.key else cur.right
        return False


# -----------------------------
# Benchmark
# -----------------------------
def avg_search_time_ns(search_fn, queries: list[int]) -> int:
    for q in queries[:50]:
        search_fn(q)

    t0 = time.perf_counter_ns()
    for q in queries:
        search_fn(q)
    t1 = time.perf_counter_ns()
    return int((t1 - t0) / len(queries))


# -----------------------------
# GUI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Benchmark: BST vs AVL vs Hash vs Arreglo Ordenado (Tkinter)")
        self.geometry("1040x680")

        self.work_q: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None

        self.txt_path = tk.StringVar(value=DEFAULT_TXT)
        self.n_var = tk.IntVar(value=DEFAULT_N)
        self.auto_var = tk.IntVar(value=DEFAULT_AUTO_SEARCHES)
        self.seed_var = tk.StringVar(value="")

        # Estructuras activas
        self.use_bst = tk.BooleanVar(value=True)
        self.use_avl = tk.BooleanVar(value=True)
        self.use_hash = tk.BooleanVar(value=True)
        self.use_sorted = tk.BooleanVar(value=True)

        # Datos
        self.data: list[int] | None = None
        self.sorted_data: list[int] | None = None
        self.hset: set[int] | None = None
        self.bst: BST | None = None
        self.avl: AVL | None = None

        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        r1 = ttk.Frame(top)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="TXT:").pack(side="left")
        ttk.Entry(r1, textvariable=self.txt_path, width=60).pack(side="left", padx=6)
        ttk.Button(r1, text="Elegir...", command=self.pick_txt).pack(side="left")

        r2 = ttk.Frame(top)
        r2.pack(fill="x", pady=6)
        ttk.Label(r2, text="N:").pack(side="left")
        ttk.Entry(r2, textvariable=self.n_var, width=12).pack(side="left", padx=6)
        ttk.Label(r2, text="Búsquedas automáticas:").pack(side="left", padx=(12, 0))
        ttk.Entry(r2, textvariable=self.auto_var, width=8).pack(side="left", padx=6)
        ttk.Label(r2, text="Seed (opcional):").pack(side="left", padx=(12, 0))
        ttk.Entry(r2, textvariable=self.seed_var, width=14).pack(side="left", padx=6)
        ttk.Label(r2, text="(vacío = aleatorio)").pack(side="left")

        r3 = ttk.Frame(top)
        r3.pack(fill="x", pady=4)
        ttk.Label(r3, text="Construir:").pack(side="left")
        ttk.Checkbutton(r3, text="BST", variable=self.use_bst).pack(side="left", padx=10)
        ttk.Checkbutton(r3, text="AVL", variable=self.use_avl).pack(side="left", padx=10)
        ttk.Checkbutton(r3, text="Tabla hash (set)", variable=self.use_hash).pack(side="left", padx=10)
        ttk.Checkbutton(r3, text="Arreglo ordenado", variable=self.use_sorted).pack(side="left", padx=10)

        r4 = ttk.Frame(top)
        r4.pack(fill="x", pady=8)
        ttk.Button(r4, text="1) Generar TXT", command=self.cmd_generate).pack(side="left")
        ttk.Button(r4, text="2) Cargar y Construir", command=self.cmd_build).pack(side="left", padx=8)
        ttk.Button(r4, text="3) Benchmark", command=self.cmd_benchmark).pack(side="left", padx=8)

        self.status = tk.StringVar(value="Listo.")
        ttk.Label(top, textvariable=self.status).pack(fill="x", pady=(6, 0))
        self.pbar = ttk.Progressbar(top, mode="determinate")
        self.pbar.pack(fill="x", pady=4)

        mid = ttk.LabelFrame(self, text="Búsqueda interactiva", padding=10)
        mid.pack(fill="x", padx=10, pady=10)

        self.query_var = tk.StringVar(value="")
        ttk.Label(mid, text="Número:").pack(side="left")
        ttk.Entry(mid, textvariable=self.query_var, width=20).pack(side="left", padx=6)
        ttk.Button(mid, text="Buscar", command=self.cmd_search_one).pack(side="left", padx=6)

        self.search_out = tk.StringVar(value="(sin búsquedas)")
        ttk.Label(mid, textvariable=self.search_out).pack(side="left", padx=12)

        bot = ttk.LabelFrame(self, text="Tabla comparativa", padding=10)
        bot.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("estructura", "avg_us", "mem", "complexity")
        self.tree = ttk.Treeview(bot, columns=cols, show="headings", height=13)
        self.tree.heading("estructura", text="Estructura")
        self.tree.heading("avg_us", text="Tiempo prom. búsqueda")
        self.tree.heading("mem", text="Memoria aprox. usada")
        self.tree.heading("complexity", text="Complejidad teórica")

        self.tree.column("estructura", width=260)
        self.tree.column("avg_us", width=180, anchor="e")
        self.tree.column("mem", width=220, anchor="e")
        self.tree.column("complexity", width=160)

        self.tree.pack(fill="both", expand=True)

    def pick_txt(self):
        path = filedialog.asksaveasfilename(
            title="Selecciona/crea TXT",
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")]
        )
        if path:
            self.txt_path.set(path)

    def _busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _start_worker(self, target):
        if self._busy():
            messagebox.showwarning("Ocupado", "Ya hay una tarea ejecutándose.")
            return
        self.worker = threading.Thread(target=target, daemon=True)
        self.worker.start()

    def _poll_queue(self):
        try:
            while True:
                msg = self.work_q.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_msg(self, msg):
        k = msg.get("kind")
        if k == "status":
            self.status.set(msg["text"])
        elif k == "progress":
            done = msg.get("done", 0)
            total = msg.get("total")
            if total is None:
                self.pbar.configure(mode="indeterminate")
                self.pbar.start(10)
            else:
                self.pbar.stop()
                self.pbar.configure(mode="determinate", maximum=total, value=done)
        elif k == "progress_stop":
            self.pbar.stop()
            self.pbar.configure(mode="determinate", maximum=1, value=0)
        elif k == "table":
            for item in self.tree.get_children():
                self.tree.delete(item)
            for row in msg["rows"]:
                self.tree.insert("", "end", values=row)
        elif k == "search_out":
            self.search_out.set(msg["text"])
        elif k == "error":
            messagebox.showerror("Error", msg["text"])

    # --------------------------
    # Acciones
    # --------------------------
    def cmd_generate(self):
        def job():
            try:
                n = int(self.n_var.get())
                if n <= 0:
                    raise ValueError("N debe ser > 0")
                path = self.txt_path.get().strip()
                if not path:
                    raise ValueError("Ruta TXT inválida")

                seed_s = self.seed_var.get().strip()
                seed = int(seed_s) if seed_s else None

                self.work_q.put({"kind": "status", "text": f"Generando {n:,} números..."})

                def prog(done, total):
                    self.work_q.put({"kind": "progress", "done": done, "total": total})
                    self.work_q.put({"kind": "status", "text": f"Generando: {done:,}/{total:,}"})

                t0 = time.perf_counter()
                generate_txt(path, n, seed, prog)
                t1 = time.perf_counter()

                self.work_q.put({"kind": "progress_stop"})
                self.work_q.put({"kind": "status", "text": f"TXT generado en {(t1 - t0):.2f} s: {path}"})
            except Exception as e:
                self.work_q.put({"kind": "progress_stop"})
                self.work_q.put({"kind": "error", "text": str(e)})

        self._start_worker(job)

    def cmd_build(self):
        def job():
            try:
                path = self.txt_path.get().strip()
                if not os.path.exists(path):
                    raise FileNotFoundError("No existe el TXT. Genera primero o elige uno válido.")

                # reset
                self.data = None
                self.sorted_data = None
                self.hset = None
                self.bst = None
                self.avl = None

                self.work_q.put({"kind": "status", "text": "Cargando TXT..."})
                self.work_q.put({"kind": "progress", "done": 0, "total": None})

                def prog(lines):
                    self.work_q.put({"kind": "status", "text": f"Cargando: {lines:,} líneas..."})

                t0 = time.perf_counter()
                data = load_txt(path, prog)
                t1 = time.perf_counter()
                self.data = data

                self.work_q.put({"kind": "progress_stop"})
                self.work_q.put({"kind": "status", "text": f"Cargados {len(data):,} números en {(t1 - t0):.2f} s"})

                # Construcciones
                if self.use_sorted.get():
                    self.work_q.put({"kind": "status", "text": "Construyendo arreglo ordenado..."})
                    self.work_q.put({"kind": "progress", "done": 0, "total": 1})
                    t0 = time.perf_counter()
                    self.sorted_data = sorted(data)
                    t1 = time.perf_counter()
                    self.work_q.put({"kind": "progress_stop"})
                    self.work_q.put({"kind": "status", "text": f"Arreglo ordenado listo en {(t1 - t0):.2f} s"})

                if self.use_hash.get():
                    self.work_q.put({"kind": "status", "text": "Construyendo tabla hash (set)..."})
                    self.work_q.put({"kind": "progress", "done": 0, "total": 1})
                    t0 = time.perf_counter()
                    self.hset = set(data)
                    t1 = time.perf_counter()
                    self.work_q.put({"kind": "progress_stop"})
                    self.work_q.put({"kind": "status", "text": f"Set listo en {(t1 - t0):.2f} s (únicos: {len(self.hset):,})"})

                if self.use_bst.get():
                    self.work_q.put({"kind": "status", "text": "Construyendo BST (puede ser MUY pesado)..."})
                    bst = BST()
                    total = len(data)

                    # Progreso por bloques
                    self.work_q.put({"kind": "progress", "done": 0, "total": total})
                    t0 = time.perf_counter()
                    for i, x in enumerate(data, 1):
                        bst.insert(x)
                        if i % 200_000 == 0:
                            self.work_q.put({"kind": "progress", "done": i, "total": total})
                            self.work_q.put({"kind": "status", "text": f"BST insert: {i:,}/{total:,} (nodos: {bst.nodes:,})"})
                    t1 = time.perf_counter()
                    self.work_q.put({"kind": "progress_stop"})
                    self.bst = bst
                    self.work_q.put({"kind": "status", "text": f"BST listo en {(t1 - t0):.2f} s (nodos distintos: {bst.nodes:,})"})

                if self.use_avl.get():
                    self.work_q.put({"kind": "status", "text": "Construyendo AVL (puede ser MUY pesado)..."})
                    avl = AVL()
                    total = len(data)

                    self.work_q.put({"kind": "progress", "done": 0, "total": total})
                    t0 = time.perf_counter()
                    for i, x in enumerate(data, 1):
                        avl.insert(x)
                        if i % 200_000 == 0:
                            self.work_q.put({"kind": "progress", "done": i, "total": total})
                            self.work_q.put({"kind": "status", "text": f"AVL insert: {i:,}/{total:,} (nodos: {avl.nodes:,})"})
                    t1 = time.perf_counter()
                    self.work_q.put({"kind": "progress_stop"})
                    self.avl = avl
                    self.work_q.put({"kind": "status", "text": f"AVL listo en {(t1 - t0):.2f} s (nodos distintos: {avl.nodes:,})"})

                self.work_q.put({"kind": "status", "text": "Construcción completada. Ya puedes buscar o correr benchmark."})

            except MemoryError:
                self.work_q.put({"kind": "progress_stop"})
                self.work_q.put({"kind": "error", "text": "MemoryError: te quedaste sin RAM (muy común con BST/AVL o set en 10M)."})
            except Exception as e:
                self.work_q.put({"kind": "progress_stop"})
                self.work_q.put({"kind": "error", "text": str(e)})

        self._start_worker(job)

    def cmd_search_one(self):
        def job():
            try:
                x = int(self.query_var.get().strip())
                parts = []

                if self.sorted_data is not None:
                    t0 = time.perf_counter_ns()
                    f = binary_search(self.sorted_data, x)
                    t1 = time.perf_counter_ns()
                    parts.append(f"Ordenado: {'SI' if f else 'NO'} ({ns_to_us(t1 - t0):.2f} us)")
                elif self.use_sorted.get():
                    parts.append("Ordenado: (no construido)")

                if self.hset is not None:
                    t0 = time.perf_counter_ns()
                    f = (x in self.hset)
                    t1 = time.perf_counter_ns()
                    parts.append(f"Hash: {'SI' if f else 'NO'} ({ns_to_us(t1 - t0):.2f} us)")
                elif self.use_hash.get():
                    parts.append("Hash: (no construido)")

                if self.bst is not None:
                    t0 = time.perf_counter_ns()
                    f = self.bst.contains(x)
                    t1 = time.perf_counter_ns()
                    parts.append(f"BST: {'SI' if f else 'NO'} ({ns_to_us(t1 - t0):.2f} us)")
                elif self.use_bst.get():
                    parts.append("BST: (no construido)")

                if self.avl is not None:
                    t0 = time.perf_counter_ns()
                    f = self.avl.contains(x)
                    t1 = time.perf_counter_ns()
                    parts.append(f"AVL: {'SI' if f else 'NO'} ({ns_to_us(t1 - t0):.2f} us)")
                elif self.use_avl.get():
                    parts.append("AVL: (no construido)")

                if not parts:
                    parts = ["No hay estructuras activas."]
                self.work_q.put({"kind": "search_out", "text": " | ".join(parts)})

            except ValueError:
                self.work_q.put({"kind": "error", "text": "Ingresa un entero válido."})
            except Exception as e:
                self.work_q.put({"kind": "error", "text": str(e)})

        self._start_worker(job)

    def cmd_benchmark(self):
        def job():
            try:
                m = int(self.auto_var.get())
                if m <= 0:
                    raise ValueError("Búsquedas automáticas debe ser > 0")

                rng = random.Random()
                queries = [rng.randint(MINV, MAXV) for _ in range(m)]

                rows = []

                # Arreglo ordenado
                if self.sorted_data is not None:
                    self.work_q.put({"kind": "status", "text": "Benchmark: arreglo ordenado..."})
                    avg_ns = avg_search_time_ns(lambda q: binary_search(self.sorted_data, q), queries)
                    mem = approx_list_memory(len(self.sorted_data))
                    rows.append(("Arreglo ordenado", f"{ns_to_us(avg_ns):.3f} us", human_bytes(mem), "O(log n)"))

                # Hash
                if self.hset is not None:
                    self.work_q.put({"kind": "status", "text": "Benchmark: tabla hash..."})
                    avg_ns = avg_search_time_ns(lambda q: (q in self.hset), queries)
                    mem = approx_set_memory(len(self.hset))
                    rows.append(("Tabla hash (set)", f"{ns_to_us(avg_ns):.3f} us", human_bytes(mem), "O(1) prom."))

                # BST
                if self.bst is not None:
                    self.work_q.put({"kind": "status", "text": "Benchmark: BST..."})
                    avg_ns = avg_search_time_ns(lambda q: self.bst.contains(q), queries)
                    # Estimación por nodo (muy variable). Usamos ~56 bytes como "idea" (subestima a veces).
                    mem = approx_tree_memory(self.bst.nodes, per_node_est=56)
                    rows.append(("BST", f"{ns_to_us(avg_ns):.3f} us", human_bytes(mem), "O(h)"))

                # AVL
                if self.avl is not None:
                    self.work_q.put({"kind": "status", "text": "Benchmark: AVL..."})
                    avg_ns = avg_search_time_ns(lambda q: self.avl.contains(q), queries)
                    # AVL node tiene un campo extra (altura): estimación ~64 bytes (solo referencia)
                    mem = approx_tree_memory(self.avl.nodes, per_node_est=64)
                    rows.append(("AVL", f"{ns_to_us(avg_ns):.3f} us", human_bytes(mem), "O(log n)"))

                if not rows:
                    raise RuntimeError("No hay estructuras construidas. Ejecuta 'Cargar y Construir'.")

                self.work_q.put({"kind": "table", "rows": rows})
                self.work_q.put({"kind": "status", "text": "Benchmark terminado (ver tabla)."})

            except Exception as e:
                self.work_q.put({"kind": "error", "text": str(e)})

        self._start_worker(job)


if __name__ == "__main__":
    App().mainloop()
