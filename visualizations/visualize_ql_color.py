import pickle
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import heapq
import traceback
# style = ttk.Style()
# style.configure('TNotebook.Tab', background="black", foreground="black")

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
        
        # Define accessible color scheme
        self.define_accessible_colors()
        
        # Apply theme and configure styles
        self.configure_styles()
        
        # Create UI elements
        self.setup_ui()
    
    def define_accessible_colors(self):
        """
        Define an accessible color scheme that works well for color blindness
        - Default mode: Using black background with white text for maximum visibility
        - Ensuring high contrast 
        - Avoiding problematic combinations (red/green)
        """
        # Default to high contrast mode with black background and white text
        self.bg_color = "#000000"  # Black background
        self.text_color = "#FFFFFF"  # White text
        self.highlight_color = "#FFFFFF"  # White (changed from yellow)
        self.secondary_color = "#444444"  # Dark gray for alternate rows
        self.contrast_color = "#333333"  # Darker gray for contrast elements
        
        # Score colors - all white text now
        self.score_colors = [
            "#FFFFFF",  # White
            "#FFFFFF",  # White
            "#FFFFFF",  # White
            "#FFFFFF",  # White
            "#FFFFFF",  # White
        ]
        
        # Set high contrast mode to enabled by default
        self.high_contrast_var = None  # Will be properly initialized in setup_ui
        
        # Set theme colors for the application
        self.root.configure(bg=self.bg_color)
        
    def configure_styles(self):
        """Configure ttk styles with accessible colors"""
        self.style = ttk.Style()
        
        # Get font size based on text size setting
        self.font_sizes = {
            "Small": 9,
            "Medium": 11,
            "Large": 14,
            "Extra Large": 18
        }
        
        # Use Medium as default if not yet set
        font_size = self.font_sizes.get(
            getattr(self, 'text_size_var', tk.StringVar(value="Medium")).get(), 
            11
        )
        
        # Configure the main theme with high contrast
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', 
                           background=self.bg_color, 
                           foreground=self.text_color, 
                           font=('Arial', font_size))
        
        # Fix button styling - ensure white text on colored background
        self.style.configure('TButton', 
                           background=self.highlight_color, 
                           foreground=self.text_color,  # Always white text on buttons 
                           font=('Arial', font_size, 'bold'))
        
        # Scale up all spinboxes and entry fields for better accessibility
        self.style.configure('TSpinbox', 
                           font=('Arial', font_size))
        
        self.style.configure('TEntry', 
                           font=('Arial', font_size))
        
        # Header style with larger font
        self.style.configure('Header.TLabel', 
                           font=('Arial', font_size + 2, 'bold'), 
                           background=self.bg_color, 
                           foreground=self.highlight_color)
        
        # Selected item style
        self.style.configure('Selected.TFrame', 
                           background=self.highlight_color)
        
        # Score highlighting styles for different ranges
        for i, color in enumerate(self.score_colors):
            # For score labels - ensure text is always visible against background
            self.style.configure(f'Score{i}.TLabel', 
                               foreground=color,  # Always text in score color
                               background=self.bg_color,  # Always on bg_color background
                               font=('Arial', font_size, 'bold'))
            
            # For score bars (frames)
            self.style.configure(f'Score{i}.TFrame', 
                               background=color)
        
        # Notebook style with clear visual distinction
        self.style.configure('TNotebook', 
                           background=self.bg_color)
        
        self.style.configure('TNotebook.Tab', 
                           font=('Arial', font_size, 'bold'),
                           padding=[10, 4],
                           background=self.highlight_color,
                           foreground=self.text_color)  # Always white text on tabs
        
        # Highlight active tab with strong visual distinction
        self.style.map('TNotebook.Tab',
                     background=[('selected', self.highlight_color), 
                                ('!selected', self.contrast_color)],
                     foreground=[('selected', self.text_color), 
                                ('!selected', self.text_color)])  # Always white text
                                
        # Consistent focus highlighting for all widgets
        self.style.map('TButton', 
                     background=[('active', self.highlight_color)],
                     foreground=[('active', self.text_color)],  # Always white text
                     relief=[('active', 'sunken')])
                     
        # Canvas elements
        for canvas in [getattr(self, 'docs_canvas', None)]:
            if canvas:
                canvas.configure(background=self.bg_color)
    
    def update_text_size(self, *args):
        """Update font sizes based on text size selection"""
        # Reconfigure styles with new font sizes
        self.configure_styles()
        
        # Update the display if data is loaded
        if self.data:
            self.update_display()
    
    def setup_ui(self):
        # Top frame for controls
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # Load data button - use tk.Button with black background and white text
        load_btn = tk.Button(control_frame, 
                           text="Load Pickle File", 
                           command=self.load_data,
                           bg="#000000",
                           fg="#FFFFFF",
                           font=('Arial', 11, 'bold'),
                           relief=tk.RAISED,
                           borderwidth=2)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Hyperparam selection
        self.hyperparam_label = ttk.Label(control_frame, text="Hyperparameter (lambda):")
        self.hyperparam_label.pack(side=tk.LEFT, padx=5)
        
        # Create a frame for the slider and markers
        slider_frame = ttk.Frame(control_frame)
        slider_frame.pack(side=tk.LEFT, padx=5, fill=tk.X)
        
        # Slider with markers
        self.hyperparam_slider = ttk.Scale(slider_frame, orient=tk.HORIZONTAL, length=200, 
                                          command=self.on_hyperparam_change)
        self.hyperparam_slider.pack(side=tk.TOP, fill=tk.X)
        
        # Frame for markers
        self.markers_frame = ttk.Frame(slider_frame)
        self.markers_frame.pack(side=tk.TOP, fill=tk.X)
        
        self.hyperparam_value_label = ttk.Label(control_frame, text="Value: None")
        self.hyperparam_value_label.pack(side=tk.LEFT, padx=5)
        
        # Query selection
        self.query_label = ttk.Label(control_frame, text="Query ID:")
        self.query_label.pack(side=tk.LEFT, padx=5)
        
        # Create a separate variable to track the selected query
        self.query_var = tk.StringVar()
        
        # Modified: Use tk.OptionMenu with black background and white text
        self.query_combo = tk.OptionMenu(control_frame, self.query_var, "")
        self.query_combo.config(width=25, 
                              bg="#000000", 
                              fg="#FFFFFF",
                              font=('Arial', 11),
                              activebackground="#333333",
                              activeforeground="#FFFFFF")
        # Also configure the dropdown menu colors
        self.query_combo['menu'].config(bg="#000000", fg="#FFFFFF")
        self.query_combo.pack(side=tk.LEFT, padx=5)
        self.query_var.trace_add("write", self.on_query_change_var)
        
        # Number of documents to display
        self.top_n_label = ttk.Label(control_frame, text="Top N docs:")
        self.top_n_label.pack(side=tk.LEFT, padx=5)
        
        self.top_n_spinbox = ttk.Spinbox(control_frame, from_=1, to=50, width=5)
        self.top_n_spinbox.set(10)
        self.top_n_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Number of words to display
        self.top_words_label = ttk.Label(control_frame, text="Top N words:")
        self.top_words_label.pack(side=tk.LEFT, padx=5)
        
        self.top_words_spinbox = ttk.Spinbox(control_frame, from_=1, to=100, width=5)
        self.top_words_spinbox.set(20)
        self.top_words_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Refresh button - black background and white text
        refresh_btn = tk.Button(control_frame, 
                              text="Refresh", 
                              command=self.update_display,
                              bg="#000000",
                              fg="#FFFFFF",
                              font=('Arial', 11, 'bold'),
                              relief=tk.RAISED,
                              borderwidth=2)
        refresh_btn.pack(side=tk.LEFT, padx=5)
        
        # Debug button - black background and white text
        debug_btn = tk.Button(control_frame, 
                            text="Debug Info", 
                            command=self.show_debug_info,
                            bg="#000000",
                            fg="#FFFFFF",
                            font=('Arial', 11, 'bold'),
                            relief=tk.RAISED,
                            borderwidth=2)
        debug_btn.pack(side=tk.LEFT, padx=5)
        
        # High contrast mode toggle (default to enabled)
        self.high_contrast_var = tk.BooleanVar(value=True)
        high_contrast_check = tk.Checkbutton(
            control_frame, 
            text="High Contrast Mode (White on Black)", 
            variable=self.high_contrast_var,
            command=self.toggle_high_contrast,
            bg=self.bg_color,
            fg=self.text_color,
            selectcolor=self.bg_color,
            font=('Arial', 11)
        )
        high_contrast_check.pack(side=tk.LEFT, padx=15)
        
        # Text size control for accessibility
        self.text_size_label = ttk.Label(control_frame, text="Text Size:")
        self.text_size_label.pack(side=tk.LEFT, padx=5)
        
        self.text_size_var = tk.StringVar(value="Medium")
        text_size_combo = ttk.Combobox(
            control_frame,
            textvariable=self.text_size_var,
            values=["Small", "Medium", "Large", "Extra Large"],
            width=10,
            state="readonly"
        )
        text_size_combo.pack(side=tk.LEFT, padx=5)
        self.text_size_var.trace_add("write", self.update_text_size)
        
        # Query Text Frame (between controls and main content)
        self.query_text_frame = ttk.LabelFrame(self.root, text="Query", padding="10")
        self.query_text_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.query_text_label = ttk.Label(self.query_text_frame, text="", wraplength=1180)
        self.query_text_label.pack(fill=tk.X, expand=True)
        
        # Main frame for visualizations
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left frame for document scores
        self.docs_frame = ttk.LabelFrame(self.main_frame, text="Document Scores", padding="10")
        self.docs_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create a frame with scrollbars for document scores
        self.docs_scroll_frame = ttk.Frame(self.docs_frame)
        self.docs_scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        self.docs_canvas = tk.Canvas(self.docs_scroll_frame, background=self.bg_color)
        self.docs_scrollbar = ttk.Scrollbar(self.docs_scroll_frame, orient="vertical", command=self.docs_canvas.yview)
        self.docs_scrollable_frame = ttk.Frame(self.docs_canvas)
        
        self.docs_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.docs_canvas.configure(scrollregion=self.docs_canvas.bbox("all"))
        )
        
        self.docs_canvas.create_window((0, 0), window=self.docs_scrollable_frame, anchor="nw")
        self.docs_canvas.configure(yscrollcommand=self.docs_scrollbar.set)
        
        self.docs_canvas.pack(side="left", fill="both", expand=True)
        self.docs_scrollbar.pack(side="right", fill="y")
        
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
    
    def toggle_high_contrast(self):
        """Toggle between regular and high contrast modes"""
        if self.high_contrast_var.get():
            # High contrast: Black background, white text for maximum visibility
            self.bg_color = "#000000"  # Black
            self.text_color = "#FFFFFF"  # White
            self.highlight_color = "#FFFFFF"  # White (changed from yellow)
            self.secondary_color = "#444444"  # Dark gray for alternate rows
            self.contrast_color = "#333333"  # Darker gray
            
            # Update score colors - all white now
            self.score_colors = [
                "#FFFFFF",  # White
                "#FFFFFF",  # White
                "#FFFFFF",  # White
                "#FFFFFF",  # White
                "#FFFFFF",  # White
            ]
        else:
            # Regular contrast: White background, dark text
            self.bg_color = "#FFFFFF"  # White
            self.text_color = "#000000"  # Black
            self.highlight_color = "#000000"  # Black
            self.secondary_color = "#EEEEEE"  # Light gray for alternate rows
            self.contrast_color = "#F0F0F0"  # Light gray
            
            # Black text for non-high-contrast mode
            self.score_colors = [
                "#000000",  # Black
                "#000000",  # Black
                "#000000",  # Black
                "#000000",  # Black
                "#000000",  # Black
            ]
        
        # Reconfigure styles with new colors
        self.configure_styles()
        
        # Force redraw of elements
        self.root.update()
        
        # Refresh the display if data is loaded
        if self.data:
            self.update_display()
    
    def toggle_controls(self, enabled):
        """Fix to properly handle enabling/disabling controls"""
        if enabled:
            # Enable controls
            self.hyperparam_slider.configure(state="normal")
            self.query_combo.configure(state="normal")
            self.top_n_spinbox.configure(state="normal")
            self.top_words_spinbox.configure(state="normal")
        else:
            # Disable controls
            self.hyperparam_slider.configure(state="disabled")
            self.query_combo.configure(state="disabled")
            self.top_n_spinbox.configure(state="disabled")
            self.top_words_spinbox.configure(state="disabled")
    
    def update_query_dropdown(self, query_ids, current_query_id=None):
        """Update the query dropdown menu with new values and select current_query_id if provided"""
        # Clear the current menu
        self.query_combo['menu'].delete(0, 'end')
        
        # Add new options
        for query_id in query_ids:
            self.query_combo['menu'].add_command(
                label=query_id,
                command=lambda q=query_id: self.query_var.set(q)
            )
        
        # Set the current selection
        if current_query_id and current_query_id in query_ids:
            self.query_var.set(current_query_id)
        elif query_ids:
            self.query_var.set(query_ids[0])
        else:
            self.query_var.set("")
    
    def create_slider_markers(self):
        # Clear previous markers
        for widget in self.markers_frame.winfo_children():
            widget.destroy()
        
        if not self.hyperparams:
            return
        
        # Create markers for the slider
        num_markers = min(10, len(self.hyperparams))  # Limit to a reasonable number
        
        # Calculate marker positions
        step = len(self.hyperparams) / (num_markers - 1) if num_markers > 1 else 0
        
        for i in range(num_markers):
            idx = min(int(i * step), len(self.hyperparams) - 1)
            value = self.hyperparams[idx]
            
            # Create marker with better visibility
            marker = ttk.Label(self.markers_frame, 
                              text=f"{value}", 
                              style='Header.TLabel')
            marker.place(relx=i/(num_markers-1) if num_markers > 1 else 0.5, anchor=tk.N)
    
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
            
            with open(filepath, 'rb') as f:
                self.data = pickle.load(f)
            
            # Print data structure for debugging
            print(f"Data type: {type(self.data)}")
            
            if not isinstance(self.data, dict):
                messagebox.showerror("Error", f"Loaded data is not a dictionary. Type: {type(self.data)}")
                return
            
            if not self.data:
                messagebox.showerror("Error", "Loaded dictionary is empty")
                return
            
            # Extract hyperparameters and sort them
            self.hyperparams = sorted(list(self.data.keys()))
            print(f"Found hyperparams: {self.hyperparams}")
            
            if not self.hyperparams:
                messagebox.showerror("Error", "No hyperparameters found in the dictionary")
                return
            
            # Configure slider
            self.hyperparam_slider.configure(from_=0, to=len(self.hyperparams)-1)
            self.hyperparam_slider.set(0)
            self.current_hyperparam = self.hyperparams[0]
            self.hyperparam_value_label.configure(text=f"Value: {self.current_hyperparam}")
            
            # Create markers for the slider
            self.create_slider_markers()
            
            # Get query IDs for the first hyperparam
            if not isinstance(self.data[self.current_hyperparam], dict):
                messagebox.showerror("Error", f"Data for hyperparam {self.current_hyperparam} is not a dictionary")
                return
            
            self.query_ids = sorted(list(self.data[self.current_hyperparam].keys()))
            print(f"Found query IDs: {self.query_ids[:5]}... (total: {len(self.query_ids)})")
            
            if not self.query_ids:
                messagebox.showerror("Error", f"No query IDs found for hyperparam {self.current_hyperparam}")
                return
            
            # First enable controls
            self.toggle_controls(True)
            
            # Then update the query dropdown
            self.update_query_dropdown(self.query_ids)
            
            # Set current query ID
            self.current_query_id = self.query_ids[0]
            
            self.update_display()
            self.status_var.set(f"Loaded {len(self.hyperparams)} hyperparameters with {len(self.query_ids)} queries.")
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
            self.status_var.set(f"Error loading file: {str(e)}")
            self.toggle_controls(False)
    
    def show_debug_info(self):
        if not self.data:
            messagebox.showinfo("Debug Info", "No data loaded yet.")
            return
        
        try:
            debug_info = "Data Structure:\n"
            debug_info += f"- Type: {type(self.data)}\n"
            debug_info += f"- Hyperparams: {self.hyperparams}\n"
            debug_info += f"- Current Hyperparam: {self.current_hyperparam}\n"
            debug_info += f"- Number of Queries: {len(self.query_ids)}\n"
            debug_info += f"- Current Query ID: {self.current_query_id}\n"
            debug_info += f"- Query Variable Value: {self.query_var.get()}\n"
            debug_info += f"- Query Combo State: {self.query_combo['state']}\n"
            
            if self.current_hyperparam is not None and self.current_query_id is not None:
                # Check if the current_query_id exists in the data
                if self.current_query_id in self.data[self.current_hyperparam]:
                    query_data = self.data[self.current_hyperparam][self.current_query_id]
                    debug_info += f"- Number of Docs for Current Query: {len(query_data)}\n"
                    
                    if query_data:
                        # Sample one document
                        sample_doc_id = next(iter(query_data))
                        sample_doc = query_data[sample_doc_id]
                        debug_info += f"- Sample Doc ID: {sample_doc_id}\n"
                        debug_info += f"- Sample Doc Keys: {sample_doc.keys()}\n"
                        
                        # Check for query_text
                        if 'query_text' in sample_doc:
                            debug_info += f"- Query text: {sample_doc['query_text'][:50]}...\n"
                        
                        # Check for documents_text
                        if 'documents_text' in sample_doc:
                            debug_info += f"- Document text: {sample_doc['documents_text'][:50]}...\n"
                        
                        # Sample word scores
                        if 'word_scores' in sample_doc and sample_doc['word_scores']:
                            word_scores = sample_doc['word_scores']
                            debug_info += f"- Sample Word Scores: {list(word_scores.items())[:3]}\n"
                else:
                    debug_info += f"- ERROR: Current query ID {self.current_query_id} not found in data for hyperparam {self.current_hyperparam}\n"
                    debug_info += f"- Available query IDs: {list(self.data[self.current_hyperparam].keys())[:5]}...\n"
            
            # Create debug window with black background and white text
            debug_window = tk.Toplevel(self.root)
            debug_window.title("Debug Information")
            debug_window.geometry("800x600")
            debug_window.configure(bg="#000000")
            
            # Create text widget with black background and white text
            text_widget = tk.Text(debug_window, wrap=tk.WORD, 
                                bg="#000000", 
                                fg="#FFFFFF",
                                font=("Arial", 11))
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(text_widget, command=text_widget.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.config(yscrollcommand=scrollbar.set)
            
            # Insert debug info
            text_widget.insert(tk.END, debug_info)
            
        except Exception as e:
            traceback.print_exc()
            messagebox.showerror("Error", f"Error generating debug info: {str(e)}")
    
    def on_hyperparam_change(self, event):
        try:
            idx = int(float(self.hyperparam_slider.get()))
            if 0 <= idx < len(self.hyperparams):
                # Store the previous query ID before changing hyperparam
                previous_query_id = self.current_query_id
                
                # Update current hyperparam
                self.current_hyperparam = self.hyperparams[idx]
                self.hyperparam_value_label.configure(text=f"Value: {self.current_hyperparam}")
                
                # Update query IDs for this hyperparam
                self.query_ids = sorted(list(self.data[self.current_hyperparam].keys()))
                
                # Debug print
                print(f"Updated query IDs for hyperparam {self.current_hyperparam}: {self.query_ids[:5]}... (total: {len(self.query_ids)})")
                
                # Update the query dropdown with the new values, trying to keep the same query selected
                self.update_query_dropdown(self.query_ids, previous_query_id)
                
                # Set current query ID (either the previous one if available or the first one)
                if previous_query_id in self.query_ids:
                    self.current_query_id = previous_query_id
                else:
                    self.current_query_id = self.query_ids[0] if self.query_ids else None
                
                self.update_display()
        except Exception as e:
            traceback.print_exc()
            self.status_var.set(f"Error changing hyperparam: {str(e)}")
    
    def on_query_change_var(self, *args):
        """Handle query selection change from the StringVar"""
        try:
            # Get selected query from the variable
            selected_query = self.query_var.get()
            
            print(f"Query selection changed to: {selected_query}")
            
            if selected_query in self.query_ids:
                self.current_query_id = selected_query
                print(f"Set current query ID to: {self.current_query_id}")
                self.update_display()
            else:
                print(f"ERROR: Selected query {selected_query} not in query_ids")
                # Debug info
                print(f"query_ids contains: {self.query_ids[:5]}...")
        except Exception as e:
            traceback.print_exc()
            self.status_var.set(f"Error changing query: {str(e)}")
    
    def get_score_style(self, score, max_score):
        """Return style name based on score relative to max score"""
        if max_score == 0:
            return 'Score2.TLabel'  # Middle style if all scores are 0
            
        ratio = score / max_score
        if ratio < 0.2:
            return 'Score0.TLabel'
        elif ratio < 0.4:
            return 'Score1.TLabel'
        elif ratio < 0.6:
            return 'Score2.TLabel'
        elif ratio < 0.8:
            return 'Score3.TLabel'
        else:
            return 'Score4.TLabel'
    
    def update_display(self, event=None):
        if not self.data or not self.current_hyperparam or not self.current_query_id:
            return
        
        try:
            # Clear existing widgets in doc scores frame
            for widget in self.docs_scrollable_frame.winfo_children():
                widget.destroy()
            
            # Clear existing widgets in word scores frame
            for widget in self.word_scores_frame.winfo_children():
                widget.destroy()
            
            # Get documents for current query
            query_data = self.data[self.current_hyperparam][self.current_query_id]
            
            if not query_data:
                self.status_var.set(f"No document data found for query {self.current_query_id}")
                return
            
            # Try to find query text from the first document (should be the same for all docs)
            query_text = ""
            for doc_id in query_data:
                if 'query_text' in query_data[doc_id]:
                    query_text = query_data[doc_id]['query_text']
                    break
            
            # Update query text display
            if query_text:
                # Highlight the query text with the header style
                self.query_text_label.config(
                    text=f"ID: {self.current_query_id} | Text: {query_text}",
                    style='Header.TLabel'
                )
            else:
                self.query_text_label.config(
                    text=f"ID: {self.current_query_id} | Text: ",
                    style='Header.TLabel'
                )
            
            try:
                top_n = int(self.top_n_spinbox.get())
                top_words = int(self.top_words_spinbox.get())
            except ValueError:
                top_n = 10
                top_words = 20
            
            # Sort documents by score
            doc_scores = []
            for doc_id, doc_data in query_data.items():
                if 'score' in doc_data:
                    doc_scores.append((doc_id, doc_data['score']))
            
            if not doc_scores:
                self.status_var.set("No document scores found")
                return
            
            top_docs = heapq.nlargest(top_n, doc_scores, key=lambda x: x[1])
            
            # Get max score for relative styling
            max_score = max(score for _, score in top_docs) if top_docs else 0
            
            # Display document scores table
            self.display_doc_scores_table(top_docs, max_score)
            
            # Display word scores for the top document
            if top_docs:
                top_doc_id = top_docs[0][0]
                self.display_word_scores(top_doc_id, top_words)
                
            self.status_var.set(f"Displaying top {len(top_docs)} documents for query {self.current_query_id} with hyperparam {self.current_hyperparam}")
            
        except Exception as e:
            traceback.print_exc()
            self.status_var.set(f"Error updating display: {str(e)}")
            messagebox.showerror("Error", f"Error updating display: {str(e)}")
    
    def display_doc_scores_table(self, top_docs, max_score):
        # Table headers
        headers = ["#", "Doc ID", "Score", "Rank", "Document Text"]
        for i, header in enumerate(headers):
            label = ttk.Label(self.docs_scrollable_frame, text=header, style='Header.TLabel')
            label.grid(row=0, column=i, padx=5, pady=2, sticky='w')
        
        # Add a separator line
        separator = ttk.Separator(self.docs_scrollable_frame, orient='horizontal')
        separator.grid(row=1, column=0, columnspan=5, sticky='ew', pady=5)
        
        # Table rows with alternating background for better readability
        for i, (doc_id, score) in enumerate(top_docs, 2):  # Start from row 2 because of the separator
            doc_data = self.data[self.current_hyperparam][self.current_query_id][doc_id]
            
            # Create row frame with alternating background
            row_frame = ttk.Frame(self.docs_scrollable_frame)
            row_frame.grid(row=i, column=0, columnspan=5, sticky='ew')
            
            # Set row index with consistent styling
            idx_label = ttk.Label(row_frame, text=f"{i-1}")
            idx_label.grid(row=0, column=0, padx=5, pady=2, sticky='w')
            
            # Make document ID clickable to show word scores - black button with white text
            doc_btn = tk.Button(
                row_frame, 
                text=doc_id,
                command=lambda d=doc_id: self.display_word_scores(d, int(self.top_words_spinbox.get())),
                bg="#000000",
                fg="#FFFFFF",
                font=('Arial', 11),
                relief=tk.RAISED,
                borderwidth=1
            )
            doc_btn.grid(row=0, column=1, padx=5, pady=2, sticky='w')
            
            # Style score based on value (relative to max)
            score_style = self.get_score_style(score, max_score)
            score_label = ttk.Label(
                row_frame, 
                text=f"{score:.4f}",
                style=score_style
            )
            score_label.grid(row=0, column=2, padx=5, pady=2, sticky='w')
            
            # Add a visual bar to represent score (for better visual cue beyond just color)
            score_frame = ttk.Frame(row_frame)
            score_frame.grid(row=0, column=2, padx=(45, 5), pady=2, sticky='w')
            
            # Calculate bar width based on score
            bar_width = int(100 * (score / max_score)) if max_score > 0 else 0
            bar = tk.Frame(score_frame, width=bar_width, height=10, bg="#FFFFFF")  # White bar
            bar.pack(side='left')
            
            # Rank information with improved formatting
            rank_text = f"{doc_data['ranks']}" if 'ranks' in doc_data else ""
            rank_label = ttk.Label(row_frame, text=rank_text)
            rank_label.grid(row=0, column=3, padx=5, pady=2, sticky='w')
            
            # Add document text with improved wrapping
            doc_text = doc_data.get('documents_text', '')
            doc_text_label = ttk.Label(row_frame, text=doc_text, wraplength=500)
            doc_text_label.grid(row=0, column=4, padx=5, pady=2, sticky='w')
            
            # Add separator between rows for better visibility
            if i < len(top_docs) + 1:  # Don't add after the last row
                row_separator = ttk.Separator(self.docs_scrollable_frame, orient='horizontal')
                row_separator.grid(row=i+1, column=0, columnspan=5, sticky='ew')
    
    def display_word_scores(self, doc_id, top_n=20):
        doc_data = self.data[self.current_hyperparam][self.current_query_id][doc_id]
        
        # Clear existing widgets
        for widget in self.word_scores_frame.winfo_children():
            widget.destroy()
        
        # Add a header showing which document we're viewing
        header_frame = tk.Frame(self.word_scores_frame, bg="#000000")
        header_frame.pack(fill=tk.X, pady=5)
        
        # Document ID header with black background and white text
        doc_header = tk.Label(
            header_frame, 
            text=f"Document ID: {doc_id}",
            font=("Arial", 12, "bold"),
            bg="#000000",
            fg="#FFFFFF"
        )
        doc_header.pack(side=tk.LEFT)
        
        # Create notebook for word scores and missing word scores
        notebook = ttk.Notebook(self.word_scores_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Word scores tab - with black background
        word_scores_tab = tk.Frame(notebook)
        notebook.add(word_scores_tab, text="Word Scores")
        
        # Missing word scores tab - with black background
        missing_word_scores_tab = tk.Frame(notebook)
        notebook.add(missing_word_scores_tab, text="Missing Word Scores")
        # tab_ids = notebook.tabs()
        # for tab_id in tab_ids:
        #     notebook.tab(tab_id, foreground="black")

        # Display word scores
        if 'word_scores' in doc_data and doc_data['word_scores']:
            self.create_word_scores_table(word_scores_tab, doc_data['word_scores'], top_n)
        else:
            tk.Label(
                word_scores_tab, 
                text="No word scores available for this document",
                bg="#000000",
                font=("Arial", 10)
            ).pack(pady=20)
        
        # Display missing word scores
        if 'missing_word_scores' in doc_data and doc_data['missing_word_scores']:
            self.create_word_scores_table(missing_word_scores_tab, doc_data['missing_word_scores'], top_n)
        else:
            tk.Label(
                missing_word_scores_tab, 
                text="No missing word scores available for this document",
                font=("Arial", 10)
            ).pack(pady=20)
    
    def create_word_scores_table(self, parent, word_scores, top_n=20):
        if not word_scores:
            no_data_label = ttk.Label(parent, text="No word scores available")
            no_data_label.pack(pady=20)
            return
        
        # Create a frame with scrollbars
        scroll_frame = ttk.Frame(parent)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(scroll_frame, background="#000000")
        scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Get max score for relative styling
        max_score = max(abs(score) for score in word_scores.values()) if word_scores else 0
        
        # Table headers with improved styling
        headers = ["#", "Word", "Score"]
        for i, header in enumerate(headers):
            header_label = ttk.Label(scrollable_frame, text=header, style='Header.TLabel')
            header_label.grid(row=0, column=i, padx=5, pady=2, sticky='w')
        
        # Add a separator after headers
        separator = ttk.Separator(scrollable_frame, orient='horizontal')
        separator.grid(row=1, column=0, columnspan=len(headers), sticky='ew', pady=5)
        
        # Sort words by score (descending absolute value)
        sorted_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Take top N words if there are more
        display_words = sorted_words[:top_n]
        
        # Table rows
        for i, (word, score) in enumerate(display_words, 1):
            # Row index (with 1-based numbering)
            row_idx = i + 1  # +1 for the separator
            
            # Row index
            index_label = tk.Label(
                scrollable_frame, 
                text=f"{i}",
                bg="#000000",
                fg="#FFFFFF",
                font=("Arial", 10)
            )
            index_label.grid(row=row_idx, column=0, padx=5, pady=2, sticky='w')
            
            # Word - make the font bold for emphasis
            word_label = tk.Label(
                scrollable_frame, 
                text=word, 
                font=("Arial", 10, "bold"),
                bg="#000000",
                fg="#FFFFFF"
            )
            word_label.grid(row=row_idx, column=1, padx=5, pady=2, sticky='w')
            
            # Score
            score_label = tk.Label(
                scrollable_frame, 
                text=f"{score:.4f}",
                bg="#000000",
                fg="#FFFFFF",
                font=("Arial", 10)
            )
            score_label.grid(row=row_idx, column=2, padx=5, pady=2, sticky='w')
            
            # Add a separator between rows
            if i < len(display_words):
                row_separator = ttk.Separator(scrollable_frame, orient='horizontal')
                row_separator.grid(row=row_idx+1, column=0, columnspan=len(headers), sticky='ew')

def main():
    root = tk.Tk()
    app = QueryDocumentVisualizer(root)
    root.mainloop()

if __name__ == "__main__":
    main()