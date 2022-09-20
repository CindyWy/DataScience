#final calculator

def calculator():

    print("\n-------------------------------------------------")
    try:
      x = float(input("Enter 1st number : "))
      y = float(input("Enter 2nd number : "))
    except ValueError:
      print("Invalid input. Please try again!\n")
      return

    print("\nPlease select the operation type")
    print("1.Add (+)")
    print("2.Subtract (-)")
    print("3.Multiply (X)")
    print("4.Divide (/)")

    choice = input("Enter your choice of calculation: ").lower()

    def add(x, y):
        return x + y

    def subtract(x, y):
        return x - y

    def multiply(x, y):
        return x * y

    def divide(x, y):
        try:
          return x / y
        except ZeroDivisionError:
          print("Undefined/None becuse not allowed to divide by zero")

    if choice in ["1","a","add","satu","tambah","+"]:
      print("==>", x, "+", y, "=", (add(x,y)))
    elif choice in ["2","s","subtract","dua","kurang","-"]:
      print("==>", x, "-", y, "=", (subtract(x,y)))
    elif choice in ["3","m","multiply","tiga","kali","x"]:
      print("==>", x, "x", y, "=", (multiply(x,y)))
    elif choice in ["4","d","divide","empat","bagi","/"]:
      print("==>", x, "/", y, "=", (divide(x,y)))
    else:
      print("That is not a valid choice.\n")

while True:
  calculator()
  next_calculation = input("\nAre you ready for next calculation? (yes/no) ").lower()
  print("\n")
  if next_calculation in ["no","tidak","stop","berhenti","exit","n"]:
    break