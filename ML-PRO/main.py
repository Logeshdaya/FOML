import warnings
warnings.filterwarnings("ignore")

print("1. Create Dataset")
print("2. Train Face Model")
print("3. Start Attendance")

choice = input("Enter Choice: ")

if choice == "1":
    import create_dataset
elif choice == "2":
    import face_train

elif choice == "3":
    import attendance
else:
    print("Invalid Choice!")
