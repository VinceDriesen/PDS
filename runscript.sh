#!/usr/bin/env bash
set -e

# Mapjes
BUILD_DIR="build"
BIN_DIR="$BUILD_DIR/bin"

# Bouw de executables
cmake -S . -B $BUILD_DIR
cmake --build $BUILD_DIR

echo "Beschikbare oefeningen:"
for exe in "$BIN_DIR"/*; do
  name=$(basename "$exe")
  echo " - $name"
done

# Vraag de gebruiker welke oefening ze willen draaien
echo -n "Voer de naam van de oefening in die je wilt uitvoeren: "
read oef

if [[ -x "$BIN_DIR/$oef" ]]; then
  echo "Start $oef..."
  "$BIN_DIR/$oef"
else
  echo "Oefening '$oef' niet gevonden!"
  exit 1
fi
