#!/bin/bash

# Replace this with your actual bucket URL
BUCKET_URL="gs://dai-ultra-research-public"

# Help function to display available commands
show_help() {
    echo "GCS Bucket Helper Script"
    echo "Usage: ./gcs-helper.sh [command] [args]"
    echo ""
    echo "Commands:"
    echo "  cpm [dataset] [model]         - Copy model from bucket"
    echo "  cpd [dataset]                 - Copy dataset from bucket"
    echo "  cpr [dataset]                 - Copy results from bucket"
    echo "  upl [dir]                     - Upload directory to bucket"
    echo "  url               - Display the bucket URL"
    echo "  help              - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./gcs-helper.sh ls"
    echo "  ./gcs-helper.sh cp local_file.txt \$BUCKET_URL/remote_path/"
    echo "  ./gcs-helper.sh cp \$BUCKET_URL/remote_file.txt local_file.txt"
}

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    echo "Error: gsutil is not installed. Please install Google Cloud SDK first."
    exit 1
fi

echo "$1"

# Handle commands
case "$1" in
    "cpm")
        if [ -z "$2" ] || [ -z "$3" ]; then
            echo "Error: cpm requires dataset and model arguments"
            show_help
            exit 1
        fi
        gsutil -m cp -r "$BUCKET_URL/probcal/$2/$3/best_loss.ckpt" "chkp/$2/$3/best_loss.ckpt"
        ;;
    "cpd")
        if [ -z "$2" ]; then
            echo "Error: cpd requires dataset argument"
            show_help
            exit 1
        fi
        gsutil -m cp -r "$BUCKET_URL/hosted-datasets/$2/*" "data/$2/"
        ;;
    "cpr")
        if [ -z "$2" ]; then
            echo "Error: cpr requires dataset argument"
            show_help
            exit 1
        fi
        mkdir -p "results/coco-people/$2"
        gsutil -m cp "$BUCKET_URL/probcal/results/coco-people/coco_$2/calibration_results.pt" "results/coco-people/$2/calibration_results.pt"
        ;;
    "upl")
        if [ -z "$2" ]; then
            echo "Error: upl requires directory argument"
            show_help
            exit 1
        fi
        gsutil -m cp -r "$2/*" "$BUCKET_URL/"
        ;;
    "url")
        echo "$BUCKET_URL"
        ;;
    "help"|*)
        show_help
        ;;
esac
