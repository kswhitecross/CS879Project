import pickle
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import heapq
from collections import defaultdict

class QueryDocumentVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Query-Document Score Visualizer")
        self.root.geometry("1200x800")
        self.data = None
        self.current_hyperparam = None
        self.hyperparams = []
        self.current_query_id = None
        self.query_ids = []
        
        # Create UI elements
        self.setup_ui()
    
    def setup_ui(self):
        # Top frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Load data button
        load_btn = ttk.Button(control_frame, text="Load Pickle File", command=self.load_data)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Hyperparam selection
        self.hyperparam_label = ttk.Label(control_frame, text="Hyperparameter:")
        self.hyperparam_label.pack(side=tk.LEFT, padx=5)
        
        self.hyperparam_slider = ttk.Scale(control_frame, orient=tk.HORIZONTAL, length=200, 
                                           command=self.on_hyperparam_change)
        self.hyperparam_slider.pack(side=tk.LEFT, padx=5)
        
        self.hyperparam_value_label = ttk.Label(control_frame, text="Value: None")
        self.hyperparam_value_label.pack(side=tk.LEFT, padx=5)
        
        # Query selection
        self.query_label = ttk.Label(control_frame, text="Query ID:")
        self.query_label.pack(side=tk.LEFT, padx=5)
        
        self.query_combo = ttk.Combobox(control_frame, state="readonly")
        self.query_combo.pack(side=tk.LEFT, padx=5)
        self.query_combo.bind("<<ComboboxSelected>>", self.on_query_change)
        
        # Number of documents to display
        self.top_n_label = ttk.Label(control_frame, text="Top N docs:")
        self.top_n_label.pack(side=tk.LEFT, padx=5)
        
        self.top_n_spinbox = ttk.Spinbox(control_frame, from_=1, to=20, width=5)
        self.top_n_spinbox.set(5)
        self.top_n_spinbox.pack(side=tk.LEFT, padx=5)
        self.top_n_spinbox.bind("<<Increment>>", self.update_display)
        self.top_n_spinbox.bind("<<Decrement>>", self.update_display)
        self.top_n_spinbox.bind("<Return>", self.update_display)
        
        # Refresh button
        refresh_btn = ttk.Button(control_frame, text="Refresh", command=self.update_display)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Main frame for visualizations
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for scores visualization
        self.scores_frame = ttk.LabelFrame(self.main_frame, text="Document Scores", padding="10")
        self.scores_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right frame for word scores
        self.word_scores_frame = ttk.LabelFrame(self.main_frame, text="Word Scores", padding="10")
        self.word_scores_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Please load a pickle file.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Disable controls initially
        self.toggle_controls(False)
    
    def toggle_controls(self, enabled):
        state = "normal" if enabled else "disabled"
        self.hyperparam_slider.state([state])
        self.query_combo.state([state.replace("normal", "readonly") if enabled else state])
    
    def load_data(self):
        filepath = filedialog.askopenfilename(
            title="Select Pickle File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            self.status_var.set(f"Loading data from {os.path.basename(filepath)}...")
            self.root.update()
            
            with open("../notebooks/ql_dict.pkl", 'rb') as f:
                self.data = pickle.load(f)
            
            # Extract hyperparameters and sort them
            self.hyperparams = sorted(list(self.data.keys()))
            
            # Configure slider
            self.hyperparam_slider.configure(from_=0, to=len(self.hyperparams)-1)
            self.hyperparam_slider.set(0)
            self.current_hyperparam = self.hyperparams[0]
            self.hyperparam_value_label.configure(text=f"Value: {self.current_hyperparam}")
            
            # Get query IDs for the first hyperparam
            self.query_ids = sorted(list(self.data[self.current_hyperparam].keys()))
            self.query_combo['values'] = self.query_ids
            
            if self.query_ids:
                self.query_combo.current(0)
                self.current_query_id = self.query_ids[0]
            
            self.toggle_controls(True)
            self.update_display()
            self.status_var.set(f"Loaded {len(self.hyperparams)} hyperparameters with {len(self.query_ids)} queries.")
            
        except Exception as e:
            self.status_var.set(f"Error loading file: {str(e)}")
            self.toggle_controls(False)
    
    def on_hyperparam_change(self, event):
        idx = int(float(self.hyperparam_slider.get()))
        if 0 <= idx < len(self.hyperparams):
            self.current_hyperparam = self.hyperparams[idx]
            self.hyperparam_value_label.configure(text=f"Value: {self.current_hyperparam}")
            
            # Update query IDs for this hyperparam
            self.query_ids = sorted(list(self.data[self.current_hyperparam].keys()))
            self.query_combo['values'] = self.query_ids
            
            if self.query_ids:
                self.query_combo.current(0)
                self.current_query_id = self.query_ids[0]
            
            self.update_display()
    
    def on_query_change(self, event):
        self.current_query_id = self.query_combo.get()
        self.update_display()
    
    def update_display(self, event=None):
        if not self.data or not self.current_hyperparam or not self.current_query_id:
            return
        
        # Clear existing widgets
        for widget in self.scores_frame.winfo_children():
            widget.destroy()
        
        for widget in self.word_scores_frame.winfo_children():
            widget.destroy()
        
        try:
            top_n = int(self.top_n_spinbox.get())
        except ValueError:
            top_n = 5
        
        # Get documents for current query
        query_data = self.data[self.current_hyperparam][self.current_query_id]
        
        # Sort documents by score
        doc_scores = [(doc_id, doc_data['score']) for doc_id, doc_data in query_data.items()]
        top_docs = heapq.nlargest(top_n, doc_scores, key=lambda x: x[1])
        
        # Plot document scores
        self.plot_doc_scores(top_docs)
        
        # Plot word scores for the top document
        if top_docs:
            top_doc_id = top_docs[0][0]
            self.plot_word_scores(top_doc_id)
    
    def plot_doc_scores(self, top_docs):
        fig, ax = plt.subplots(figsize=(6, 4))
        
        doc_ids = [doc_id for doc_id, _ in top_docs]
        scores = [score for _, score in top_docs]
        
        y_pos = np.arange(len(doc_ids))
        
        ax.barh(y_pos, scores, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(doc_ids)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Score')
        ax.set_title(f'Top Document Scores for Query {self.current_query_id}')
        
        canvas = FigureCanvasTkAgg(fig, master=self.scores_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add detailed information table
        info_frame = ttk.Frame(self.scores_frame)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Table headers
        headers = ["Doc ID", "Score", "Rank"]
        for i, header in enumerate(headers):
            label = ttk.Label(info_frame, text=header, font=("", 10, "bold"))
            label.grid(row=0, column=i, padx=5, pady=2, sticky='w')
        
        # Table rows
        for i, (doc_id, score) in enumerate(top_docs, 1):
            doc_data = self.data[self.current_hyperparam][self.current_query_id][doc_id]
            
            ttk.Label(info_frame, text=doc_id).grid(row=i, column=0, padx=5, pady=2, sticky='w')
            ttk.Label(info_frame, text=f"{score:.4f}").grid(row=i, column=1, padx=5, pady=2, sticky='w')
            
            rank_text = f"{doc_data['ranks']}" if 'ranks' in doc_data else "N/A"
            ttk.Label(info_frame, text=rank_text).grid(row=i, column=2, padx=5, pady=2, sticky='w')
    
    def plot_word_scores(self, doc_id):
        doc_data = self.data[self.current_hyperparam][self.current_query_id][doc_id]
        
        # Create notebook for word scores and missing word scores
        notebook = ttk.Notebook(self.word_scores_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Word scores tab
        if 'word_scores' in doc_data and doc_data['word_scores']:
            self.create_word_score_tab(notebook, "Word Scores", doc_data['word_scores'], doc_id)
        
        # Missing word scores tab
        if 'missing_word_scores' in doc_data and doc_data['missing_word_scores']:
            self.create_word_score_tab(notebook, "Missing Word Scores", doc_data['missing_word_scores'], doc_id)
    
    def create_word_score_tab(self, notebook, title, word_scores, doc_id):
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=title)
        
        # Sort words by score
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 20 words if there are more
        display_words = sorted_words[:20]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        words = [word for word, _ in display_words]
        scores = [score for _, score in display_words]
        
        y_pos = np.arange(len(words))
        
        ax.barh(y_pos, scores, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Score')
        ax.set_title(f'{title} for Doc {doc_id}')
        
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    root = tk.Tk()
    app = QueryDocumentVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()