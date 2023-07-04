import argparse
import sys

from Utils.fileconverter import itk_to_nrrd
from Engine.train import initiate as train_initiate
from Engine.evaluate import initiate as evaluate_initiate
from Utils.MSD_utilities import generate_training_data
from Utils.plot import plot_train_data
from Utils.PETCT_LungDx_utilities import convert_nsclc_petct


class Parser(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Interface for utilities',
            usage='''interface <command> [<args>]

        Available commands:
        itk_to_nrrd             Converts files in folder from other ITK formats to .nrrd format
        train                   Initiates training
        evaluate                Initiates evaluation
        plot                    Generates and shows plot based on specified metric file
        msd_generate_data_file  Generates a datafile for training on MSD dataset
        format_petct_dataset    Converts the the LUNGPTCT dataset to nifti files and generates segboxes for each scan

        ''')
        parser.add_argument('command', help='Subcommand to run')

        args = parser.parse_args(sys.argv[1:2])
        if(not hasattr(self, args.command)):
            print('Unrecognized command')
            parser.print_help()
            exit(1)
        
        getattr(self, args.command)()

    def itk_to_nrrd(self):
        parser = argparse.ArgumentParser(
            description='Converts files in folder from other ITK formats to .nrrd format')

        parser.add_argument("source")
        parser.add_argument("destination")

        args = parser.parse_args(sys.argv[2:])
        itk_to_nrrd(args.source, args.destination)

    def train(self):
        parser = argparse.ArgumentParser(
            description='Initiates training of model based on specified config file')
        parser.add_argument("config")
        args = parser.parse_args(sys.argv[2:])
        train_initiate(args.config)

    def evaluate(self):
        parser = argparse.ArgumentParser(
            description='Initiates evaluation of model based on specified config file')
        parser.add_argument("config")
        args = parser.parse_args(sys.argv[2:])
        evaluate_initiate(args.config)

    def plot(self):
        parser = argparse.ArgumentParser(
            description='Generates and shows plot based on specified metric file')
        parser.add_argument("metric_file")
        parser.add_argument("-store_file")
        parser.add_argument("--dont_show", "-d", default=False)
        parser.add_argument("-steps_per_epoch", default=-1)
        args = parser.parse_args(sys.argv[2:])
        
        plot_train_data(args.metric_file, args.store_file, not args.dont_show, steps_in_epoch=int(args.steps_per_epoch))

    def msd_generate_data_file(self):
        parser = argparse.ArgumentParser(
            description='Generates a datafile for training on MSD dataset')
        parser.add_argument("datafile_save_path")
        parser.add_argument("msd_directory")
        parser.add_argument("target_platform")
        args = parser.parse_args(sys.argv[2:])
        generate_training_data(args.datafile_save_path, args.msd_directory, args.target_platform)

    def format_petct_dataset(self):
        parser = argparse.ArgumentParser(
            description='Converts the the LUNGPTCT dataset to nifti files and generates segboxes for each scan')
        parser.add_argument("annotation_folder")
        parser.add_argument("dataset_folder")
        parser.add_argument("label_folder")
        parser.add_argument("input_folder")
        args = parser.parse_args(sys.argv[2:])
        convert_nsclc_petct(args.annotation_folder, args.dataset_folder, args.label_folder, args.input_folder)

if __name__ == '__main__':
    Parser()