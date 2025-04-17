import threading
import tkinter as tk
from tkinter import filedialog
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageTk
from sklearn.decomposition import IncrementalPCA
import plotly.offline as pyo
import plotly.graph_objects as go
import tempfile
import webbrowser

@tf.keras.utils.register_keras_serializable(name="euclidean_distance")
def euclidean_distance(twins):
    x, y = twins
    return keras.ops.norm(x - y, axis=1, keepdims=True)

@tf.keras.utils.register_keras_serializable(name="contrastive_loss")
def contrastive_loss(y, d):
    margin = 1.0
    y = keras.ops.cast(y, d.dtype)
    return (1 - y)/2 * keras.ops.square(d) + y/2 * keras.ops.square(keras.ops.maximum(0.0, margin - d))

def create_mnist_pairs(X, y, num_pairs=1000):
    pairs, labels = [], []
    indices = [np.where(y==i)[0] for i in range(10)]
    for _ in range(num_pairs // 2):
        # positive
        c = np.random.randint(0,10)
        i1, i2 = np.random.choice(indices[c], 2, False)
        pairs.append([X[i1], X[i2]]); labels.append(0)
        # negative
        c1, c2 = np.random.choice(10,2,False)
        pairs.append([X[np.random.choice(indices[c1])], X[np.random.choice(indices[c2])]])
        labels.append(1)
    return np.array(pairs), np.array(labels)

class SiameseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Siamese Network Explorer")

        # Models & caches
        self.model = None
        self.embedding_model = None
        self._mnist_data = None
        self._pairs = None
        self._pca = None
        self._prototypes = None

        # Current images & embeddings
        self.image1 = self.image2 = None
        self.emb1 = self.emb2 = None

        self._build_ui()

    def _build_ui(self):
        mf = tk.LabelFrame(self.root, text="Model", padx=5, pady=5)
        mf.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(mf, text="Load Model", command=self._safe(self.load_model)).pack(side=tk.LEFT)
        self.status = tk.Label(mf, text="No model loaded"); self.status.pack(side=tk.LEFT, padx=10)

        ix = tk.LabelFrame(self.root, text="Images", padx=5, pady=5)
        ix.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(ix, text="Load Image 1", command=self._safe(lambda: self.load_image(1))).grid(row=0, column=0)
        tk.Button(ix, text="Load Image 2", command=self._safe(lambda: self.load_image(2))).grid(row=0, column=1)
        self.lbl1 = tk.Label(ix); self.lbl1.grid(row=1, column=0, padx=5)
        self.lbl2 = tk.Label(ix); self.lbl2.grid(row=1, column=1, padx=5)

        af = tk.LabelFrame(self.root, text="Analysis", padx=5, pady=5)
        af.pack(fill=tk.X, padx=10, pady=5)
        tk.Button(af, text="Classify Image 1", command=self._safe(self.classify_image)).pack(side=tk.LEFT)
        tk.Button(af, text="Compare Images",  command=self._safe(self.compare_images)).pack(side=tk.LEFT, padx=10)
        tk.Button(af, text="Show 3D Plot",     command=self._safe(self.show_3d_plot)).pack(side=tk.LEFT)

        tk.Label(af, text="Points:").pack(side=tk.LEFT, padx=5)
        self.points_entry = tk.Entry(af, width=6)
        self.points_entry.insert(0, "1000")
        self.points_entry.pack(side=tk.LEFT)

        self.result = tk.Label(af, text="Results will appear here")
        self.result.pack(side=tk.LEFT, padx=10)

    def _safe(self, fn):
        def wrapper(*a, **k):
            threading.Thread(target=lambda: self._exec(fn, *a, **k), daemon=True).start()
        return wrapper

    def _exec(self, fn, *a, **k):
        try:
            fn(*a, **k)
        except Exception as e:
            print(e)
            self.result.config(text=f"Failed: {e}", fg="red")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Keras model","*.h5 *.keras")])
        if not path: return
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={
                'euclidean_distance': euclidean_distance,
                'contrastive_loss': contrastive_loss
            }
        )
        self.embedding_model = self.model.layers[2]
        self.status.config(text=f"Model loaded: {path.split('/')[-1]}")

    def load_image(self, n):
        path = filedialog.askopenfilename()
        if not path: return
        img = Image.open(path).convert('L').resize((28,28))
        arr = np.array(img)/255.0
        photo = ImageTk.PhotoImage(img.resize((100,100)))
        if n == 1:
            self.image1, self.emb1 = arr, None
            self.lbl1.config(image=photo); self.lbl1.image = photo
        else:
            self.image2, self.emb2 = arr, None
            self.lbl2.config(image=photo); self.lbl2.image = photo

    def preprocess(self, img):
        return img.reshape(1, 784)

    def classify_image(self):
        if self.model is None: raise ValueError("Please load a model first")
        if self.image1 is None: raise ValueError("Please load Image 1 first")

        if self._prototypes is None:
            (X_train, y_train), _ = keras.datasets.mnist.load_data()
            X_train = X_train.reshape(-1,784).astype("float32")/255.0
            centroids = []
            for digit in range(10):
                idx = np.where(y_train==digit)[0]
                sel = np.random.choice(idx, min(200, len(idx)), replace=False)
                emb_batch = self.embedding_model.predict(X_train[sel], batch_size=64)
                centroids.append(emb_batch.mean(axis=0))
            self._prototypes = np.vstack(centroids)

        emb = self.embedding_model.predict(self.preprocess(self.image1), batch_size=1)[0]
        self.emb1 = emb
        dists = np.linalg.norm(self._prototypes - emb, axis=1)
        pred = int(dists.argmin()); dist = float(dists.min())

        self.result.config(
            text=f"Predicted digit: {pred}    (dist {dist:.3f})",
            fg="black"
        )

    def compare_images(self):
        if self.model is None: raise ValueError("Please load a model first")
        if self.image1 is None or self.image2 is None: raise ValueError("Please load both images first")

        imgs = np.vstack([self.preprocess(self.image1), self.preprocess(self.image2)])
        embs = self.embedding_model.predict(imgs, batch_size=64)
        d = np.linalg.norm(embs[0] - embs[1])
        sim = d < 0.5

        color = "green" if sim else "red"
        self.result.config(
            text=f"Distance: {d:.3f}\nSimilar: {'YES' if sim else 'NO'}",
            fg=color
        )

    def show_3d_plot(self):
        try:
            num_pts = int(self.points_entry.get())
        except ValueError:
            num_pts = 1000
            self.result.config(text="Invalid points, using 1000", fg="red")

        # load & cache MNIST
        if self._mnist_data is None:
            (_, _), (X, y) = keras.datasets.mnist.load_data()
            X = X.reshape(-1, 784).astype("float32") / 255.0
            self._mnist_data = (X, y)
        X, y = self._mnist_data

        # cache pairs
        if self._pairs is None:
            self._pairs = create_mnist_pairs(X, y, num_pairs=20000)
        pairs, labels = self._pairs

        flat = pairs.reshape(-1, 784)
        emb_all = self.embedding_model.predict(flat, batch_size=256)
        N = len(pairs)
        emb1 = emb_all[:N]
        emb2 = emb_all[N:2*N]
        pair_emb = np.vstack([emb1, emb2])
        pair_lbl = np.hstack([labels, labels])

        pts = min(num_pts, X.shape[0])
        test_emb = self.embedding_model.predict(X[:pts], batch_size=256)
        test_lbl = y[:pts]

        all_emb = np.vstack([pair_emb, test_emb])
        all_lbl = np.hstack([pair_lbl, test_lbl])
        if self._pca is None:
            ipca = IncrementalPCA(n_components=3, batch_size=1024)
            self._pca = ipca.fit(all_emb)
        pcs = self._pca.transform(all_emb)

        # build Plotly figure with legend
        fig = go.Figure()

        offset = len(pair_emb)

        # Test images (colored by digit label) with hover showing the digit
        fig.add_trace(go.Scatter3d(
            x=pcs[offset:offset+pts, 0],
            y=pcs[offset:offset+pts, 1],
            z=pcs[offset:offset+pts, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=test_lbl,
                colorscale='Rainbow',
                opacity=0.7,
                colorbar=dict(title='Digit Class')
            ),
            customdata=test_lbl,
            hovertemplate="Digit: %{customdata}<extra></extra>",
            name='Test Images'
        ))

        # Highlight current images
        for emb, col, name in [
            (self.emb1, 'red', 'Current Image 1'),
            (self.emb2, 'blue', 'Current Image 2')
        ]:
            if emb is not None:
                p = self._pca.transform(emb.reshape(1, -1))[0]
                fig.add_trace(go.Scatter3d(
                    x=[p[0]], y=[p[1]], z=[p[2]],
                    mode='markers',
                    marker=dict(size=8, color=col),
                    hovertemplate=f"{name}<extra></extra>",
                    name=name
                ))

        # layout with legend
        fig.update_layout(
            legend=dict(
                title='Legend',
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='left',
                x=0
            ),
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2',
                zaxis_title='PC3'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=700
        )

        fn = tempfile.NamedTemporaryFile(suffix='.html', delete=False).name
        pyo.plot(fig, filename=fn, auto_open=False)
        webbrowser.open(f'file://{fn}')

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x400")
    SiameseGUI(root)
    root.mainloop()