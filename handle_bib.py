"""
BibTeX Processing Script for Validating and Formatting Bibliographies

This script processes a BibTeX file, assigns standardized citation keys,
validates required fields for each entry type, and outputs a cleaned version
for integration into a collective book.

### Features:
1. **Load and Parse BibTeX Files**: Reads a BibTeX file and extracts entries.
2. **Validate Mandatory Fields**:
   - Ensures each entry contains the required fields based on its type.
   - Removes extra fields and retains only necessary ones.
   - Reports entries that do not conform to expected formats.
3. **Standardize Citation Keys**:
   - Assigns a consistent citation key to each entry.
   - Resolves duplicates using a predefined format.
4. **Filter and Save Processed Entries**:
   - Writes corrected entries to an output BibTeX file (`src/formatted.bib`).
   - Reports problematic entries that require manual correction.

### Usage:
Please replace the file src/references.bib by your own bib file.
After running the script, please use the src/formatted.bib for your references in the shared overleaf.
If some references cannot be solved, add their keys in the white_list variable.
"""
import bibtexparser
from functools import reduce

mandatory_fields = {
"article" : [{"author", "title", "journal", "volume","number", "pages", "year"},
             {"author", "title", "journal", "volume", "pages", "year"},],
"book" : [{"author", "title", "publisher", "year"},
          { "editor", "title", "publisher", "year"}],
"booklet" : [{"author", "title", "year"}],
"inbook" : [{ "author", "title", "year", "edition", "publisher", "chapter"},
            { "author", "title", "year", "publisher", "chapter"},
            { "author", "title", "year", "publisher", "pages"},
            { "author", "title", "year", "edition", "publisher", "pages"}],
"incollection" : [{ "author", "title", "pages", "editor", "booktitle", "publisher", "year"},
                  {"author", "title", "crossref", "pages"}],
"inproceedings" : [{"author", "title", "pages", "editor", "booktitle", "year", "address", "publisher"},
                   {"author", "title", "pages", "booktitle", "year", "address", "publisher"},
                   { "crossref", "author", "title", "pages"}],
"manual" : [{ "author", "title", "organization", "year"}],
"mastersthesis" : [ {"author", "title", "school", "year"}],
"misc" : [{ "author", "title", "howpublished", "year"}],
"phdthesis" : [{ "author", "title", "school", "year"}],
"proceedings" : [{ "editor", "title", "booktitle", "address", "year", "publisher"}],
"techreport" : [{ "author", "title", "institution", "number", "year"}],
}


def load_bibtex(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return bibtexparser.load(f)


def check_mandatory_fields(bib_database, mandatory_fields):
    invalid_entries = []
    new_entries = []
    for i, entry in enumerate(bib_database.entries):
        entry_key = entry.get("ID", "UNKNOWN")
        new_key = generate_alpha_label(entry)
        entry_type = entry.get("ENTRYTYPE", "UNKNOWN")
        fields = set(entry.keys()) - {"ENTRYTYPE", "ID"}
        for k in list(entry.keys()):
            if entry[k] == "":
                del entry[k]


        potential_fields_lists = mandatory_fields.get(entry_type.lower(), [[]])
        allowed_fields = set(reduce(set.union, potential_fields_lists)).union({"ENTRYTYPE", "ID"})
        new_entry = {key: value for key, value in entry.items() if key in allowed_fields}
        new_entry["ID"] = new_key
        new_entries.append(new_entry)
        fields = set(new_entry.keys()) - {"ENTRYTYPE", "ID"}

        if not any(fields == set(potential_fields) for potential_fields in potential_fields_lists):
            invalid_entries.append((entry_key, entry_type, fields))

    bib_database.entries = new_entries
    return invalid_entries


def generate_alpha_label(entry):
    """Generate an alpha-style label from a BibTeX entry."""
    authors = entry.get("author", "Anon").split(" and ")
    year = entry.get("year", "0000")[-2:]  # Last two digits of the year

    # Process author names
    last_names = [a.split(",")[0].strip() for a in authors]  # Extract last names
    if len(last_names) <= 4:
        author_part = "".join(name[0] for name in last_names)  # First letter of each
    else:
        author_part = "".join(name[0] for name in last_names[:3]) + "+"  # First 3 + "+"

    return f"{author_part}{year}"



def main():
    file='book.bib'
    bib_database = load_bibtex(file)
    invalid_entries = check_mandatory_fields(bib_database, mandatory_fields)
    with open('formatted.bib', 'w') as bib_file:
        bibtexparser.dump(bib_database, bib_file)

    # Add the citation keys of the bib entries that you cannot solve. Please keep this list and transfer them to
    # Gilles so that he can manage these. The citation keys should be the new keys produced by this script.
    # example: white_list = ["SGR+23", "GMZK17", "KSZ+18"]
    white_list = [ ]

    if invalid_entries:
        print(f"Entries with incorrect fields in {file}:")
        for entry_id, entry_type, fields in invalid_entries:
            if entry_id in white_list:
                continue
            print(f"  - Entry '{entry_id}', of type {entry_type} has fields: {sorted(list(fields))}")
            expected = [sorted(x) for x in mandatory_fields.get(entry_type.lower(), [])]
            expected = " or ".join(map(str, expected))
            print(f"    Expected: {expected}")
    else:
        print("All entries have the correct fields.")

    print(len([e[0] for e in invalid_entries if e[0] not in white_list]), len(bib_database.entries))


if __name__ == "__main__":
    main()
