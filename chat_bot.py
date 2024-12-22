import re
import tkinter

import pandas as pd
import pyttsx3
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

import csv




import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

training = pd.read_csv('Data/Training.csv')
testing = pd.read_csv('Data/Testing.csv')

cols = training.columns
cols = cols[:-1]

x = training[cols]
y = training['prognosis']
y1 = y

reduced_data = training.groupby(training['prognosis']).max()

# mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)
# print(clf.score(x_train,y_train))
# print ("cross result========")

scores = cross_val_score(clf, x_test, y_test, cv=3)
# print (scores)
print(scores.mean())

model = SVC()
model.fit(x_train, y_train)

print("for svm: ")
print(model.score(x_test, y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols



severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index


def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()
class ChatbotGUI:
    def _init_(self, root, clf, cols):
        self.root = root
        self.clf = clf
        self.cols = cols
        self.root.title("Healthcare Chatbot")
        self.root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
        self.root.configure(bg="#FFFFFF")
        self.final_list = []
        self.string = ""

        # Create a Canvas widget to serve as the base layer
        self.canvas = tk.Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
        self.canvas.pack(fill=tk.BOTH, expand=True)

        try:
            bg_image = Image.open("C:\\Users\\PMLS\\PycharmProjects\\DiseasePredictionSystem\\background2.jpg")
            bg_image = bg_image.resize((self.root.winfo_screenwidth(), self.root.winfo_screenheight()),
                                       Image.Resampling.LANCZOS)
            self.bg_photo = ImageTk.PhotoImage(bg_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.bg_photo)
        except Exception as e:
            print(f"Error loading background image: {e}")

        # Define the dimensions for the frames
        left_frame_width = root.winfo_screenwidth() // 3 - 50  # Adjusted width
        self.left_frame = tk.Frame(self.canvas, bg="#FFFFFF", width=left_frame_width)
        self.left_frame.place(x=10, y=10, width=left_frame_width,
                              height=root.winfo_screenheight() - 20)  # Slightly reduced height

        right_frame_width = root.winfo_screenwidth() - (left_frame_width + 40)  # Adjusted width to balance both sides
        self.right_frame = tk.Frame(self.canvas, bg="#F8F9FA", width=right_frame_width)
        self.right_frame.place(x=left_frame_width + 20, y=10, width=right_frame_width,
                               height=root.winfo_screenheight() - 20)  # Slightly reduced height

        # Load and display an image in the left frame
        try:
            image = Image.open("C:\\Users\\PMLS\\PycharmProjects\\DiseasePredictionSystem\\injection.jpg")  # Replace with your image path
            image = image.resize((left_frame_width, self.root.winfo_screenheight()), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.img_label = tk.Label(self.left_frame, image=self.photo, bg="#FFFFFF")
            self.img_label.pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            print(f"Error loading image: {e}")

        # Variables to store input values
        self.name_var = tk.StringVar()  # Variable for name
        self.symptom_var = tk.StringVar()  # Variable for symptom
        self.number_var = tk.StringVar()  # Variable for number
        self.days_var = tk.StringVar()  # Variable for days

        # Name input
        self.name_label = tk.Label(self.right_frame, text="Name:", font=("Arial", 14), bg="#F8F9FA")
        self.name_label.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="w")

        self.name_entry = tk.Entry(self.right_frame, font=("Arial", 14), width=30, textvariable=self.name_var)
        self.name_entry.grid(row=0, column=1, pady=10)

        # Symptom input
        self.symptom_label = tk.Label(self.right_frame, text="Enter Symptom:", font=("Arial", 14), bg="#F8F9FA")
        self.symptom_label.grid(row=1, column=0, padx=(0, 10), pady=10, sticky="w")

        self.symptom_entry = tk.Entry(self.right_frame, font=("Arial", 14), width=30, textvariable=self.symptom_var)
        self.symptom_entry.grid(row=1, column=1, pady=10)



        # Submit button (placed next to the symptom input box)
        self.submit_button = tk.Button(self.right_frame, text="Submit", font=("Arial", 14), bg="#007BFF",
                                       fg="#FFFFFF", command=lambda: self.save_input(self.symptom_entry, self.symptom_var, action='submit'))
        self.submit_button.grid(row=1, column=2, padx=(10, 0), pady=10)


        # Error message box (initially hidden)
        self.error_label = tk.Label(self.right_frame, text="", font=("Arial", 12), fg="red", bg="#F8F9FA")
        self.error_label.grid(row=2, column=0, columnspan=3, pady=5, sticky="w")

        # Scrollable chat display (text box and input layout)
        self.search_label = tk.Label(self.right_frame, text="Searches related to input:", font=("Arial", 14),
                                     bg="#F8F9FA")
        self.search_label.grid(row=3, column=0, padx=(0, 10), pady=10, sticky="w")

        # Scrollable text box
        self.canvas = tk.Canvas(self.right_frame, bg="#F8F9FA")
        self.scrollbar = tk.Scrollbar(self.right_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#F8F9FA")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=3, column=1, pady=10, sticky="nsew")
        self.scrollbar.grid(row=3, column=2, pady=10, sticky="ns")

        # Text display in scrollable frame
        self.chat_display = tk.Text(self.scrollable_frame, font=("Arial", 12), bg="#FFFFFF", wrap=tk.WORD,
                                    state='disabled', height=20, width=42)
        self.chat_display.pack(fill=tk.BOTH, expand=True)


        # Input box for search (next to scrollable text box)
        self.search_input = tk.Entry(self.right_frame, font=("Arial", 14), width=30)
        self.search_input.grid(row=4, column=1, padx=10, pady=10, sticky="w")

        # OK button to save input value (next to the input box)
        self.ok_button = tk.Button(self.right_frame, text="OK", font=("Arial", 14), bg="#28A745", fg="#FFFFFF", command=lambda: self.save_input(self.search_input, self.number_var, action='ok'))
        self.ok_button.grid(row=4, column=2, padx=10, pady=10)

        # Label for entering number from above
        self.search_label = tk.Label(self.right_frame, text="Enter number from above:", font=("Arial", 14),
                                     bg="#F8F9FA")
        self.search_label.grid(row=4, column=0, padx=(0, 10), pady=10, sticky="w")

        # Label for how many days
        self.search_label_days = tk.Label(self.right_frame, text="Okay, for how many days?:", font=("Arial", 14),
                                          bg="#F8F9FA")
        self.search_label_days.grid(row=5, column=0, padx=(0, 10), pady=10, sticky="w")

        # Input box for how many days
        self.search_days = tk.Entry(self.right_frame, font=("Arial", 14), width=30)
        self.search_days.grid(row=5, column=1, padx=10, pady=10, sticky="w")

        # OK button to save the input days
        self.ok_days = tk.Button(self.right_frame, text="OK", font=("Arial", 14), bg="#28A745", fg="#FFFFFF",
                                   command=lambda: self.save_input(self.search_days, self.days_var, action='days'))
        self.ok_days.grid(row=5, column=2, padx=10, pady=10)


        # next button for new window
        self.ok_next = tk.Button(self.right_frame, text="Next", font=("Arial", 20), bg="#28A745", fg="#FFFFFF",
                                   command=self.open_next_window)
        self.ok_next.grid(row=6, column=1, padx=20, pady=20)

    def tree_to_code(self,tree, feature_names, symptom_var, number_var,days_var, act):
        tree_ = tree.tree_

        feature_name = [

            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        chk_dis = ",".join(feature_names).split(",")

        symptoms_present = []
        # symptom_input = symp_inp


        disease_input = symptom_var.get()

        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:

            cnf_dis_str = [f"{num}) {it}" for num, it in enumerate(cnf_dis)]
            if act=='submit':
                self.display_text(self.chat_display, cnf_dis_str)

            # print("conf: ",cnf_dis_str)
            #
            # for num, it in enumerate(cnf_dis):
            #     print(f"{num}) {it}")

            #print(f"Select the one you meant (0 - {len(cnf_dis) - 1}): ", end="")
            if act=='ok':
                try:
                    conf_inp = int(number_var.get())  # Convert to integer to get input as a number
                except:
                    tkinter.messagebox.showerror("Invalid Input", "Please enter a valid number.")
                if conf_inp >= 0 and conf_inp < len(cnf_dis):
                    disease_input = cnf_dis[conf_inp]
                    print("disease_input: ", disease_input)
                else:
                    tkinter.messagebox.showerror("Invalid Input", "Please enter a valid number.")

            elif act=='days':
                try:
                    num_days = int(days_var.get())
                    print("num_days", num_days)

                except:
                    tkinter.messagebox.showerror("Invalid Input", "Please enter a valid days.")

        else:
            tkinter.messagebox.showerror("Invalid Input", "Please enter a valid symptom.")





        def recurse(node, depth):
            indent = "  " * depth

            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                # print("eshaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

                threshold = tree_.threshold[node]

                if name == disease_input:

                    val = 1
                else:
                    val = 0

                if val <= threshold:
                    recurse(tree_.children_left[node], depth + 1)
                else:

                    symptoms_present.append(name)
                    recurse(tree_.children_right[node], depth + 1)
            else:
                # print("adeenaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                present_disease = print_disease(tree_.value[node])
                # print( "You may have " +  present_disease )
                red_cols = reduced_data.columns
                symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]

                # Create a list from symptoms_present
                dis_list = list(symptoms_present)

                # Combine dis_list and symptoms_given into a single list
                combined_list = dis_list + list(symptoms_given)

                if len(dis_list) != 0:
                    print("symptoms present  " + str(dis_list))
                print("symptoms given " + str(list(symptoms_given)))

                # Print the combined list for reference
                print("Combined list of symptoms: " + str(combined_list))

                print("Are you experiencing any ")


                self.final_list = combined_list
                print("this is final_list of recurse:", self.final_list)

                symptoms_exp = []

                second_prediction = sec_predict(symptoms_exp)
                # print(second_prediction)
                calc_condition(symptoms_exp, num_days)
                string1=""
                string2=""


                if (present_disease[0] == second_prediction[0]):
                    string1=string1+" "+"You may have ", present_disease[0]
                    string1=string1+" "+(description_list[present_disease[0]])
                    # print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrr",description_list[present_disease[0]])
                    # readn(f"You may have {present_disease[0]}")
                    # readn(f"{description_list[present_disease[0]]}")

                else:
                    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                    string1 = "{} You may have {} or {}".format(string1, present_disease[0], second_prediction[0])
                    string1=string1+" "+(description_list[present_disease[0]])
                    string1=string1+" "+(description_list[second_prediction[0]])

                # print(description_list[present_disease[0]])
                precution_list = precautionDictionary[present_disease[0]]
                string2 = "{} \nTake following measures : \n".format(string2)
                for i, j in enumerate(precution_list):
                    string2 = string2 + f" {i + 1}) {j}\n"
                string2 = string1 + " " + string2

                print("string2 from recurse: ",string2)
                self.string=string2
                # confidence_level = (1.0*len(symptoms_present))/len(symptoms_given)
                # print("confidence level is " + str(confidence_level))

        recurse(0, 1)

    def save_input(self, text_box, target_var,action):
        # Get the input from the text box
        user_input = text_box.get()
        act=action
        print(f"User input saved in variable: {user_input}")

        # Save the input in the associated StringVar
        target_var.set(user_input)

        threading.Thread(target=self.tree_to_code, args=(self.clf, self.cols, self.symptom_var, self.number_var,self.days_var,act)).start()




    def display_text(self,text_box, data):
        # Clear the text box before displaying new content
        text_box.delete('1.0', tk.END)

        # Check if the data is a list
        if isinstance(data, list):
            # Convert list to string
            content = "\n".join(data)  # Joining list items with a newline
            print("got the list")
        else:
            # Assume data is a string
            content = data
            print(data)
            print("else")

        # Insert the content into the text box
        text_box.config(state='normal')
        text_box.insert(tk.END, content)
        print("inserted data")
        text_box.config(state='disabled')

    def open_next_window(self):
        # Create a new top-level window (new window)
        next_window = tk.Toplevel(self.root)

        # Set the size of the new window (same as the current window)
        next_window.geometry(self.root.geometry())  # Use the same geometry as the main window

        # Set the title for the new window
        next_window.title("Next Window")

        # Create a Canvas widget to display the background image
        canvas = tk.Canvas(next_window, width=next_window.winfo_screenwidth(), height=next_window.winfo_screenheight())
        canvas.pack(fill=tk.BOTH, expand=True)

        try:
            # Load and display the background image
            bg_image = Image.open("C:\\Users\\PMLS\\PycharmProjects\\DiseasePredictionSystem\\background2.jpg")
            bg_image = bg_image.resize(
                (next_window.winfo_screenwidth(), next_window.winfo_screenheight()),
                Image.Resampling.LANCZOS
            )
            bg_photo = ImageTk.PhotoImage(bg_image)
            canvas.create_image(0, 0, anchor=tk.NW, image=bg_photo)
        except Exception as e:
            print(f"Error loading background image: {e}")

        # Grid configuration: Create two columns for left and right frames
        next_window.grid_columnconfigure(0, weight=1, minsize=200)  # 1/3 width for left frame
        next_window.grid_columnconfigure(1, weight=2, minsize=400)  # 2/3 width for right frame

        # Create the left frame (1/3 of the window)
        left_frame = tk.Frame(next_window, bg="#F8F9FA")
        left_frame.place(
            x=10, y=10, width=(next_window.winfo_screenwidth() // 3) - 20, height=next_window.winfo_screenheight() - 20
        )

        # Load and display the image in the left frame
        try:
            pic_path = "C:\\Users\\PMLS\\PycharmProjects\\DiseasePredictionSystem\\injection.jpg"
            pic = Image.open(pic_path)
            pic = pic.resize((next_window.winfo_screenwidth() // 3, next_window.winfo_screenheight() - 20),
                             Image.Resampling.LANCZOS)
            picture = ImageTk.PhotoImage(pic)
            image_label = tk.Label(left_frame, image=picture, bg="#F8F9FA")
            image_label.pack(fill=tk.BOTH, expand=True)  # Position the image in the left frame
            image_label.image = picture  # Keep a reference to the image to prevent it from being garbage collected
        except Exception as e:
            print(f"Error loading image: {e}")

        # Create the right frame (2/3 of the window)
        right_frame = tk.Frame(next_window, bg="#F8F9FA")
        right_frame.place(
            x=(next_window.winfo_screenwidth() // 3) + 5, y=10,
            width=(next_window.winfo_screenwidth() * 2 // 3) - 20,
            height=next_window.winfo_screenheight() - 20
        )

        # Keep the reference to the background image
        canvas.image = bg_photo

        # Add label for "Symptoms Present" at the top of the right frame
        symptoms_present_label = tk.Label(right_frame, text="Symptoms Present:", font=("Arial", 12), bg="#F8F9FA")
        symptoms_present_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Create a Canvas to hold the checkboxes
        canvas = tk.Canvas(right_frame, bg="#F8F9FA", bd=0, highlightthickness=0)
        canvas.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Create a scrollable frame for the canvas
        canvas_frame = tk.Frame(canvas, bg="#F8F9FA")
        canvas_frame.grid(row=0, column=0, padx=10, pady=10)

        # Create a vertical scrollbar for the Canvas
        scrollbar = tk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        canvas.config(yscrollcommand=scrollbar.set)

        # List of symptoms

        symptoms_list = self.final_list

        print("got the list of symptoms:",symptoms_list)
        # Create a dictionary to hold the checkbuttons
        checkbox_vars = {}

        # Function to create checkboxes dynamically
        for idx, symptom in enumerate(symptoms_list):
            var = tk.BooleanVar()  # A Boolean variable for the checkbox state
            checkbox_vars[symptom] = var

            # Create a checkbox for each symptom and place it in the canvas_frame
            checkbox = tk.Checkbutton(canvas_frame, text=symptom, variable=var, bg="#F8F9FA", font=("Arial", 10))
            # Calculate the column based on the index (3 columns)
            column = idx % 3
            row = idx // 3
            checkbox.grid(row=row, column=column, sticky="w", padx=5, pady=5)

        # Make the canvas scrollable
        canvas.create_window((0, 0), window=canvas_frame, anchor="nw")
        canvas.configure(scrollregion=canvas.bbox("all"))

        # Button to save selected symptoms
        def save_selected_symptoms():
            selected_symptoms = []
            for symptom, var in checkbox_vars.items():
                if var.get():  # Check if the checkbox is selected (True)
                    selected_symptoms.append(symptom)





            # Print the selected symptoms (or save them as needed)

            print("Selected Symptoms:", selected_symptoms)

            print("self.string in open window:", self.string)

        def display_pre(text_box, data):
            # Clear the text box before displaying new content
            text_box.delete('1.0', tk.END)


            content = data
            print(data)

            # Insert the content into the text box
            text_box.config(state='normal')
            text_box.insert(tk.END, content)
            print("inserted data in prec")
            text_box.config(state='disabled')


        # Button to save the results
        save_button = tk.Button(right_frame, text="Save Symptoms", font=("Arial", 12), bg="#28a745",
                                command=save_selected_symptoms)
        save_button.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")




        # Add label for "Description and Precaution" at the top of the right frame
        descript_label = tk.Label(right_frame, text="Description and Precaution", font=("Arial", 12), bg="#F8F9FA")
        descript_label.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        # Create a text box below the description label
        precaution_text = tk.Text(right_frame, height=10, wrap="word", font=("Arial", 10))
        precaution_text.grid(row=5, column=0, padx=10, pady=10, sticky="nsew")

        # Add a vertical scrollbar to the text box
        scrollbar_text = tk.Scrollbar(right_frame, orient="vertical", command=precaution_text.yview)
        scrollbar_text.grid(row=5, column=1, sticky="ns", padx=5, pady=10)
        precaution_text.config(yscrollcommand=scrollbar_text.set)

        # Button to give precaution and description
        des_button = tk.Button(right_frame, text="Precautions & Description", font=("Arial", 12), bg="#28a745",
                               command=lambda: display_pre(precaution_text, self.string))
        des_button.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        # Optionally, you can disable the original window (root) to make it modal-like
        self.root.withdraw()  # Hide the original window (you can re-show it later with next_window.protocol())

        # If you want to make the next window close the previous window when it is closed:
        next_window.protocol("WM_DELETE_WINDOW", self.close_previous_window)

    def close_previous_window(self):
        # Show the previous window again when the next window is closed
        self.root.deiconify()  # Show the original window again
        self.ok_next.config(state=tk.NORMAL)  # Re-enable the "Next" button if needed

    # Instance variable to store the symptom after submission
    symptom_on_submit = ""  # Initialize the variable


def calc_condition(exp, days):

    sum = 0
    for item in exp:
        sum = sum + severityDictionary[item]

    if ((sum * days) / (len(exp) + 1) > 13):
        print("You should take the consultation from doctor. ")

    else:

        print("It might not be that bad but you should take precautions.")

def getDescription():

    global description_list

    with open('MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description = {row[0]: row[1]}

            description_list.update(_description)

def getSeverityDict():

    global severityDictionary
    with open('MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}

                severityDictionary.update(_diction)
        except:
            pass

def getprecautionDict():
    global precautionDictionary

    with open('MasterData/symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)



def check_pattern(dis_list, inp):
    pred_list = []

    inp = inp.replace(' ', '_')
    patt = f"{inp}"

    regexp = re.compile(patt)

    pred_list = [item for item in dis_list if regexp.search(item)]
    if (len(pred_list) > 0):

        return 1, pred_list
    else:
        return 0, []

def sec_predict(symptoms_exp):
    df = pd.read_csv('Data/Training.csv')

    X = df.iloc[:, :-1]

    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=20)  # random_state=20: Ensures reproducibility of the split.
    rf_clf = DecisionTreeClassifier()

    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}

    input_vector = np.zeros(len(symptoms_dict))

    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])



def print_disease(node):

    node = node[0]

    val = node.nonzero()

    disease = le.inverse_transform(val[0])

    return list(map(lambda x: x.strip(), list(disease)))


getSeverityDict()
getDescription()
getprecautionDict()

def first_window():
    welcome_root = tk.Tk()
    welcome_root.title("Healthcare Chatbot")
    welcome_root.attributes("-fullscreen", True)  # Make the window fullscreen

    # Create a Canvas widget for the background
    canvas = tk.Canvas(welcome_root, width=welcome_root.winfo_screenwidth(), height=welcome_root.winfo_screenheight())
    canvas.pack(fill=tk.BOTH, expand=True)

    try:
        # Load and display the background image
        bg_image = Image.open("C:\\Users\\PMLS\\PycharmProjects\\DiseasePredictionSystem\\background2.jpg")
        bg_image = bg_image.resize(
            (welcome_root.winfo_screenwidth(), welcome_root.winfo_screenheight()),
            Image.Resampling.LANCZOS
        )
        bg_photo = ImageTk.PhotoImage(bg_image)
        canvas.create_image(0, 0, anchor=tk.NW, image=bg_photo)
    except Exception as e:
        print(f"Error loading background image: {e}")

    # Add a label on top of the background
    label = tk.Label(canvas, text="Welcome to Healthcare Chatbot", font=("Arial", 48, "bold"), bg=None, fg="Green")
    label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Center the label on the screen

    # Speak the welcome message
    threading.Thread(target=readn, args=("Welcome to Healthcare Chatbot",)).start()

    # Close after 3 seconds and open main window
    def close_welcome():
        welcome_root.destroy()
        main_window()

    welcome_root.after(23000, close_welcome)
    welcome_root.mainloop()

# Function to Initialize Main Chat Window
def main_window():
    root = tk.Tk()
    app = ChatbotGUI(root,clf,cols)
    root.mainloop()

# Function to Start the Application
def start_app():
    # Initialize data


    # Start with the welcome window
    first_window()

if _name_ == "_main_":
    start_app()
# check_pattern()
# sec_predict()
# print_disease()





print("----------------------------------------------------------------------------------------")