import json
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.dates import date2num, num2date, DateFormatter
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from datetime import datetime

class MyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pocket Weight Calculator")

        self.style = ttk.Style()
        self.style.configure("TButton", padding=6, relief="flat", foreground="black")
        self.style.configure("TEntry", padding=6, relief="flat")
        
        self.create_widgets()
        self.load_weights_from_json()

    def create_widgets(self):
        # Input Widgets
        self.label_target_weight = ttk.Label(self.root, text="Target Weight:")
        self.entry_target_weight = ttk.Entry(self.root, width=20, style="TEntry")
        self.button_add_target_weight = ttk.Button(self.root, text="Add Target Weight", command=self.add_target_weight, width=20, style="TButton")

        self.label_current_weight = ttk.Label(self.root, text="Current Weight:")
        self.entry_current_weight = ttk.Entry(self.root, width=20, style="TEntry")
        self.label_date = ttk.Label(self.root, text="Date:")
        self.calendar = DateEntry(self.root, width=20, background='darkblue', foreground='white', borderwidth=2, date_pattern="dd.MM.yyyy", style="TEntry")
        self.button_add_weight = ttk.Button(self.root, text="Add Weight", command=self.add_weight, width=20, style="TButton")
        self.button_remove_weight = ttk.Button(self.root, text="Remove Weight", command=self.remove_weight, width=20, style="TButton")

        # Layout
        self.label_target_weight.grid(row=1, column=1, padx=10, pady=10, sticky="E")
        self.entry_target_weight.grid(row=1, column=2, padx=10, pady=10, sticky="W")
        self.button_add_target_weight.grid(row=1, column=3, padx=10, pady=10)

        self.label_current_weight.grid(row=2, column=1, padx=10, pady=10, sticky="E")
        self.entry_current_weight.grid(row=2, column=2, padx=10, pady=10, sticky="W")
        self.button_add_weight.grid(row=2, column=3, padx=10, pady=10)
        
        self.label_date.grid(row=3, column=1, padx=10, pady=10, sticky="E")
        self.calendar.grid(row=3, column=2, padx=10, pady=10, sticky="W")
        self.button_remove_weight.grid(row=3, column=3, padx=10, pady=10)

        # Matplotlib Figure and Canvas
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.plot = self.fig.add_subplot(1, 1, 1)
        self.scatter_points, = self.plot.plot([], [], marker='o', markersize=8, color='blue', label='Current Weights')
        self.plot.xaxis.set_major_formatter(DateFormatter('%d.%m'))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=1, columnspan=3)

        # Goal Text Label
        self.label_goal_text = tk.Label(self.root, text="", padx=10, pady=10)
        self.label_goal_text.grid(row=5, column=1, columnspan=3)

    def add_weight(self):
        try:
            current_weight = float(self.entry_current_weight.get())
            selected_date = self.calendar.get_date()
            selected_date_num = date2num(selected_date)

            existing_dates = self.scatter_points.get_xdata()
            if selected_date_num in existing_dates:
                index = np.where(existing_dates == selected_date_num)[0][0]
                self.scatter_points.set_ydata(index, current_weight)
            else:
                x_data, y_data = self.scatter_points.get_data()
                x_data = list(x_data) + [selected_date_num]
                y_data = list(y_data) + [current_weight]

                color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
                self.scatter_points.set_data(x_data, y_data)
                self.scatter_points.set_color(color)

            self.update_plot()
            self.save_weights_to_json()

        except ValueError:
            print("Error: Enter a valid weight value.")

    def remove_weight(self):
        try:
            selected_date = self.calendar.get_date()
            selected_date_num = date2num(selected_date)

            existing_dates = self.scatter_points.get_xdata()
            if selected_date_num in existing_dates:
                index = np.where(existing_dates == selected_date_num)[0][0]

                x_data, y_data = self.scatter_points.get_data()
                x_data = np.delete(x_data, index)
                y_data = np.delete(y_data, index)

                self.scatter_points.set_data(x_data, y_data)

                self.update_plot()
                self.save_weights_to_json()

            else:
                print("Error: No point found for the selected date.")

        except ValueError:
            print("Error: Enter a valid date.")

    def add_target_weight(self):
        try:
            target_weight = float(self.entry_target_weight.get())

            for line in self.plot.lines:
                if line.get_label() == 'Predicted Goal Achievement':
                    line.remove()

            for line in self.plot.lines:
                if line.get_label() == 'Target Weight':
                    line.remove()

            self.plot.axhline(y=target_weight, color='purple', linestyle='--', label='Target Weight')

            self.update_plot()
            self.save_weights_to_json()

        except ValueError:
            print("Error: Enter a valid target weight value.")

    def update_plot(self):
        self.plot.legend()
        self.plot.relim()
        self.plot.autoscale()
        self.canvas.draw()
        self.calculate_goal()

    def calculate_goal(self):
        existing_dates = self.scatter_points.get_xdata()
        existing_weights = self.scatter_points.get_ydata()
        target_weight = float(self.entry_target_weight.get())

        if target_weight and len(existing_weights) >= 2:
            dates_num = np.array(existing_dates).reshape(-1, 1)

            model = LinearRegression()
            model.fit(dates_num, existing_weights)

            target_date_num = (target_weight - model.intercept_) / model.coef_[0]

            regression_line_dates = np.linspace(min(existing_dates), target_date_num, 100)
            regression_line_weights = model.predict(regression_line_dates.reshape(-1, 1))

            if len(self.plot.lines) > 2:
                self.plot.lines[-1].remove()

            self.plot.plot(regression_line_dates, regression_line_weights, color='#90ee90', linestyle='--', label='Predicted Goal Achievement')

            today_date_num = date2num(datetime.now())
            days_until_goal = int(target_date_num - today_date_num)

            self.plot.legend()

            self.plot.relim()
            self.plot.autoscale()

            self.canvas.draw()

            if days_until_goal < 0:
                self.label_goal_text.config(text='', fg='black', font=('Helvetica', 10))
            else:
                self.label_goal_text.config(text=f'You will reach the goal in: {days_until_goal} day(s)', fg='black', font=('Helvetica', 10))

            if existing_weights[-1] == target_weight:
                self.label_goal_text.config(text='Goal achieved. Congratulations!', fg='green', font=('Helvetica', 10))
        else:
            for line in self.plot.lines:
                if line.get_label() == 'Predicted Goal Achievement':
                    line.remove()
                    self.plot.legend()
                    self.plot.relim()
                    self.plot.autoscale()
                    self.canvas.draw()

    def save_weights_to_json(self):
        data = {
            "target_weight": float(self.entry_target_weight.get()),
            "weights": [
                {"date": date.strftime("%Y-%m-%d"), "weight": weight}
                for date, weight in zip(map(num2date, self.scatter_points.get_xdata()), self.scatter_points.get_ydata())
            ]
        }

        with open("weights.json", "w") as json_file:
            json.dump(data, json_file, indent=2)

    def load_weights_from_json(self):
        try:
            with open("weights.json", "r") as json_file:
                data = json.load(json_file)

            self.entry_target_weight.delete(0, tk.END)
            self.entry_target_weight.insert(0, data["target_weight"])

            weights_data = [(datetime.strptime(item["date"], "%Y-%m-%d"), item["weight"]) for item in data.get("weights", [])]
            x_data, y_data = zip(*weights_data)
            self.scatter_points.set_data(date2num(x_data), y_data)

            self.update_plot()

        except FileNotFoundError:
            print("No previous weights file found.")

        except Exception as e:
            print(f"Error loading weights from JSON: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MyApp(root)
    root.mainloop()
