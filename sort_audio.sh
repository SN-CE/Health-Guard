#!/bin/bash

src=$1
trgt=$2
csv=$3

show_help() {
	echo "Usage: $0 <audio source dir> <audio target dir> <metadata.csv>"
	echo
	echo "Please enter file paths relative to current directory or enter absolute paths. Omit trailing /"
}

if [[ $# -eq 1 && ($1 == "--help" || $1 == "-h")]]; then
	show_help
	exit 0
fi

if [[ $# -ne 3 ]]; then
	echo "Arguments mismatch. Try --help"
	exit 1
fi

shopt -s nullglob

tail -n +2 "${csv}" | while IFS=, read -r no subject cohort sub label _; do
	for	file in "${src}/${subject}"*; do
		if [[ -f "${file}" ]]; then
			if [[ "${label}" == TB ]]; then
				echo "Copying ${file} from ${src} to ${trgt}/tb_positive/"
				cp "${file}" "${trgt}/tb_positive/"
			elif [[ "${label}" == NTB ]]; then
				echo "Copying ${file} from ${src} to ${trgt}/tb_negative/"
				cp "${file}" "${trgt}/tb_negative/"
			fi
		fi
	done
done
shopt -u nullglob
