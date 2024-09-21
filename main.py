import argparse
from model_merger import ModelMerger
from test_interface import TestInterface

def main():
    parser = argparse.ArgumentParser(description="Merge two large language models and test the result.")
    parser.add_argument("--merge", action="store_true", help="Merge the models")
    parser.add_argument("--test", action="store_true", help="Test the merged model")
    args = parser.parse_args()

    if args.merge:
        merger = ModelMerger()
        merger.merge_models()
    
    if args.test:
        tester = TestInterface()
        tester.run_test_interface()

if __name__ == "__main__":
    main()
