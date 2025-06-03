def main():
    """
    Main entry point for the project.
    Offers high-level control (menu/dispatch).
    """
    print("Welcome to ImageCompression!")
    print("Choose an option:")
    print("1. Run benchmark")
    print("2. Launch Streamlit app")
    print("3. Exit")

    choice = input("Enter your choice (1-3): ")

    if choice == '1':
        from scripts.run_benchmark import main as run_benchmark_main
        run_benchmark_main()

    elif choice == '2':
        import os
        os.system("streamlit run dct_streamlit_app/app.py")

    elif choice == '3':
        print("Exiting...")

    else:
        print("Invalid choice. Please select 1, 2 or 3.")


if __name__ == '__main__':
    main()