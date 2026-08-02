#!/bin/sh
set -eu

fail() {
    printf '%s\n' "ERROR $1 $2" >&2
    exit 1
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repository_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
lock_path="$repository_root/environment/bootstrap_toolchain.lock"
bootstrap_path="$repository_root/scripts/bootstrap_local_environment.py"
tools_root="$repository_root/.tools"
python_root="$tools_root/python"
cache_root="$tools_root/cache"

[ -f "$lock_path" ] || fail "FD-LOCK-MISSING" "Toolchain lock is missing: $lock_path"

case "$(uname -s)" in
    Darwin)
        platform=macos
        translated=$(sysctl -in sysctl.proc_translated 2>/dev/null || true)
        [ "$translated" != "1" ] || fail "FD-ROSETTA" "Translated Rosetta execution is unsupported; use a native terminal."
        ;;
    Linux)
        platform=linux-gnu
        command -v getconf >/dev/null 2>&1 || fail "FD-LIBC" "GNU libc could not be established."
        case "$(getconf GNU_LIBC_VERSION 2>/dev/null || true)" in
            glibc\ *) ;;
            *) fail "FD-LIBC" "Linux GNU libc is required; musl and unknown libc are unsupported."
        esac
        ;;
    *) fail "FD-OS" "Unsupported operating system: $(uname -s)" ;;
esac

case "$(uname -m)" in
    x86_64|amd64) architecture=x86_64 ;;
    arm64|aarch64) architecture=aarch64 ;;
    *) fail "FD-ARCH" "Unsupported architecture: $(uname -m)" ;;
esac

schema=
uv_version=
cpython_version=
asset_filename=
asset_sha256=
seen_windows_x86_64=
seen_windows_aarch64=
seen_macos_x86_64=
seen_macos_aarch64=
seen_linux_gnu_x86_64=
seen_linux_gnu_aarch64=

while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
        *=*) key=${line%%=*}; value=${line#*=} ;;
        *) fail "FD-LOCK-MALFORMED" "Toolchain lock contains a malformed record."
    esac
    case "$key" in
        schema)
            [ -z "$schema" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock duplicates schema."
            schema=$value
            ;;
        uv_version)
            [ -z "$uv_version" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock duplicates uv_version."
            uv_version=$value
            ;;
        cpython_version)
            [ -z "$cpython_version" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock duplicates cpython_version."
            cpython_version=$value
            ;;
        asset)
            old_ifs=$IFS
            IFS='|'
            set -- $value
            IFS=$old_ifs
            [ "$#" -eq 4 ] || fail "FD-LOCK-MALFORMED" "Toolchain lock contains a malformed asset record."
            record_platform=$1
            record_architecture=$2
            record_filename=$3
            record_sha256=$4
            case "$record_filename" in
                ''|.*|*/*|*\\*|.|..) fail "FD-LOCK-MALFORMED" "Toolchain lock contains an unsafe asset filename." ;;
            esac
            case "$record_sha256" in *[!0123456789abcdefABCDEF]*) fail "FD-LOCK-MALFORMED" "Toolchain lock contains an invalid SHA-256." ;; esac
            [ "${#record_sha256}" -eq 64 ] || fail "FD-LOCK-MALFORMED" "Toolchain lock contains an invalid SHA-256."
            case "$record_platform:$record_architecture" in
                windows:x86_64)
                    [ -z "$seen_windows_x86_64" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock duplicates windows/x86_64."
                    case "$record_filename" in *.zip) ;; *) fail "FD-LOCK-MALFORMED" "Windows uv asset must be a zip archive." ;; esac
                    seen_windows_x86_64=1
                    ;;
                windows:aarch64)
                    [ -z "$seen_windows_aarch64" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock duplicates windows/aarch64."
                    case "$record_filename" in *.zip) ;; *) fail "FD-LOCK-MALFORMED" "Windows uv asset must be a zip archive." ;; esac
                    seen_windows_aarch64=1
                    ;;
                macos:x86_64)
                    [ -z "$seen_macos_x86_64" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock duplicates macos/x86_64."
                    case "$record_filename" in *.tar.gz) ;; *) fail "FD-LOCK-MALFORMED" "macOS uv asset must be a tar.gz archive." ;; esac
                    seen_macos_x86_64=1
                    ;;
                macos:aarch64)
                    [ -z "$seen_macos_aarch64" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock duplicates macos/aarch64."
                    case "$record_filename" in *.tar.gz) ;; *) fail "FD-LOCK-MALFORMED" "macOS uv asset must be a tar.gz archive." ;; esac
                    seen_macos_aarch64=1
                    ;;
                linux-gnu:x86_64)
                    [ -z "$seen_linux_gnu_x86_64" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock duplicates linux-gnu/x86_64."
                    case "$record_filename" in *.tar.gz) ;; *) fail "FD-LOCK-MALFORMED" "Linux uv asset must be a tar.gz archive." ;; esac
                    seen_linux_gnu_x86_64=1
                    ;;
                linux-gnu:aarch64)
                    [ -z "$seen_linux_gnu_aarch64" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock duplicates linux-gnu/aarch64."
                    case "$record_filename" in *.tar.gz) ;; *) fail "FD-LOCK-MALFORMED" "Linux uv asset must be a tar.gz archive." ;; esac
                    seen_linux_gnu_aarch64=1
                    ;;
                *) fail "FD-LOCK-MALFORMED" "Toolchain lock contains an unsupported platform record." ;;
            esac
            if [ "$record_platform" = "$platform" ] && [ "$record_architecture" = "$architecture" ]; then
                asset_filename=$record_filename
                asset_sha256=$record_sha256
            fi
            ;;
        *) fail "FD-LOCK-MALFORMED" "Toolchain lock contains an unknown key: $key" ;;
    esac
done < "$lock_path"

[ "$schema" = "1" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock must use schema=1."
[ "$uv_version" = "0.12.1" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock must pin uv 0.12.1."
[ "$cpython_version" = "3.12.13" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock must pin CPython 3.12.13."
case "$uv_version" in ''|*[!0123456789.]*|.*|*.) fail "FD-LOCK-MALFORMED" "Toolchain lock contains an invalid uv version." ;; esac
case "$cpython_version" in ''|*[!0123456789.]*|.*|*.) fail "FD-LOCK-MALFORMED" "Toolchain lock contains an invalid CPython version." ;; esac
for required in "$seen_windows_x86_64" "$seen_windows_aarch64" "$seen_macos_x86_64" "$seen_macos_aarch64" "$seen_linux_gnu_x86_64" "$seen_linux_gnu_aarch64"; do
    [ "$required" = "1" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock is missing a required platform record."
done
[ -n "$asset_filename" ] || fail "FD-LOCK-MALFORMED" "Toolchain lock has no asset for this platform."

command -v tar >/dev/null 2>&1 || fail "FD-EXTRACT" "tar is required to extract the verified uv archive."
if command -v sha256sum >/dev/null 2>&1; then
    hash_file() { set -- $(sha256sum "$1"); printf '%s\n' "$1"; }
elif command -v shasum >/dev/null 2>&1; then
    hash_file() { set -- $(shasum -a 256 "$1"); printf '%s\n' "$1"; }
else
    fail "FD-CHECKSUM" "sha256sum or shasum -a 256 is required to verify uv."
fi

uv_version_root="$tools_root/uv/$uv_version"
platform_directory="$uv_version_root/$platform-$architecture"
archive_path="$platform_directory/$asset_filename"
uv_binary="$platform_directory/uv"
mkdir -p "$uv_version_root" "$python_root" "$cache_root" || fail "FD-READ-ONLY" "Repository-local tool directories cannot be created."

archive_matches() {
    [ -f "$1" ] || return 1
    actual=$(hash_file "$1") || return 1
    actual=$(printf '%s' "$actual" | tr '[:upper:]' '[:lower:]')
    expected=$(printf '%s' "$asset_sha256" | tr '[:upper:]' '[:lower:]')
    [ "$actual" = "$expected" ]
}

valid_uv_version() {
    version_output_file=$(mktemp "$uv_version_root/.version.XXXXXX" 2>/dev/null) || return 1
    if "$1" --version >"$version_output_file" 2>/dev/null; then
        :
    else
        rm -f "$version_output_file" 2>/dev/null || :
        return 1
    fi

    first_line=
    extra_line=
    if exec 3<"$version_output_file"; then
        IFS= read -r first_line <&3 || :
        if IFS= read -r extra_line <&3; then
            exec 3<&-
            rm -f "$version_output_file" 2>/dev/null || :
            return 1
        fi
        exec 3<&-
    else
        rm -f "$version_output_file" 2>/dev/null || :
        return 1
    fi
    rm -f "$version_output_file" 2>/dev/null || :

    carriage_return=$(printf '\r')
    case "$first_line" in
        *"$carriage_return"*) return 1 ;;
    esac

    expected_uv_prefix="uv $uv_version"
    case "$first_line" in
        "$expected_uv_prefix") return 0 ;;
        "$expected_uv_prefix ("*)
            metadata=${first_line#"$expected_uv_prefix ("}
            case "$metadata" in
                *')') metadata=${metadata%')'} ;;
                *) return 1 ;;
            esac
            [ -n "$metadata" ] || return 1
            case "$metadata" in
                *'('*|*')'*) return 1 ;;
                *) return 0 ;;
            esac
            ;;
        *) return 1 ;;
    esac
}

valid_local_uv() {
    archive_matches "$archive_path" || return 1
    [ -x "$uv_binary" ] || return 1
    valid_uv_version "$uv_binary"
}

if ! valid_local_uv; then
    staging_directory=$(mktemp -d "$uv_version_root/.staging.XXXXXX") || fail "FD-READ-ONLY" "A repository-local uv staging directory cannot be created."
    cleanup_staging() { rm -rf "$staging_directory"; }
    trap cleanup_staging EXIT HUP INT TERM
    staging_archive="$staging_directory/$asset_filename"
    if archive_matches "$archive_path"; then
        cp "$archive_path" "$staging_archive" || fail "FD-UV-ARCHIVE" "Verified local uv archive could not be staged."
    elif command -v curl >/dev/null 2>&1; then
        curl --fail --location --proto '=https' --tlsv1.2 --output "$staging_archive" "https://github.com/astral-sh/uv/releases/download/$uv_version/$asset_filename" || fail "FD-NETWORK" "uv archive download failed; check network, TLS, proxy, disk space, and write access."
    elif command -v wget >/dev/null 2>&1; then
        wget --https-only --secure-protocol=TLSv1_2 -O "$staging_archive" "https://github.com/astral-sh/uv/releases/download/$uv_version/$asset_filename" || fail "FD-NETWORK" "uv archive download failed; check network, TLS, proxy, disk space, and write access."
    else
        fail "FD-DOWNLOAD" "curl or wget is required to download the verified uv archive."
    fi
    archive_matches "$staging_archive" || fail "FD-CHECKSUM" "uv archive SHA-256 did not match the toolchain lock."
    mkdir "$staging_directory/extract" "$staging_directory/payload" || fail "FD-EXTRACT" "uv staging directory could not be prepared."
    tar -xzf "$staging_archive" -C "$staging_directory/extract" || fail "FD-EXTRACT" "Verified uv archive could not be extracted."
    extracted_root=${asset_filename%.tar.gz}
    staged_uv="$staging_directory/extract/$extracted_root/uv"
    [ -x "$staged_uv" ] || fail "FD-UV-EXECUTABLE" "Verified uv archive did not contain the expected executable."
    valid_uv_version "$staged_uv" || fail "FD-UV-VERSION" "Verified uv archive reported an unexpected version."
    mv "$staging_archive" "$staging_directory/payload/$asset_filename"
    mv "$staged_uv" "$staging_directory/payload/uv"
    backup_directory=
    if [ -e "$platform_directory" ]; then
        backup_directory="$uv_version_root/.previous.$$.${RANDOM:-0}"
        mv "$platform_directory" "$backup_directory" || fail "FD-UV-REPLACE" "Invalid repository-local uv directory could not be replaced."
    fi
    mv "$staging_directory/payload" "$platform_directory" || fail "FD-UV-REPLACE" "Verified uv directory could not be installed atomically."
    [ -z "$backup_directory" ] || rm -rf "$backup_directory"
    trap - EXIT HUP INT TERM
    cleanup_staging
fi

valid_uv_version "$uv_binary" || fail "FD-UV-VERSION" "Repository-local uv reported an unexpected version."

export UV_PYTHON_INSTALL_DIR="$python_root"
export UV_CACHE_DIR="$cache_root"
export UV_MANAGED_PYTHON=1
export UV_PYTHON_DOWNLOADS=manual
export UV_PYTHON_INSTALL_BIN=0
export UV_PYTHON_INSTALL_REGISTRY=0
export UV_NO_CONFIG=1

"$uv_binary" --no-config python install --no-bin "cpython@$cpython_version" || fail "FD-PYTHON-INSTALL" "Managed CPython installation failed; check network, TLS, proxy, disk space, and write access."
managed_python=$("$uv_binary" --no-config python find --managed-python "cpython@$cpython_version" 2>/dev/null) || fail "FD-PYTHON-FIND" "Exact managed CPython could not be found."
[ -x "$managed_python" ] || fail "FD-PYTHON-FIND" "Exact managed CPython path is not executable."

python_check='from pathlib import Path; import sys; root=Path(sys.argv[1]).resolve(); paths=(Path(sys.base_prefix).resolve(), Path(sys.executable).resolve()); raise SystemExit(0 if sys.version_info[:3] == (3, 12, 13) and sys.implementation.name == "cpython" and all(path.is_relative_to(root) for path in paths) else 1)'
venv_check='from pathlib import Path; import sys; root=Path(sys.argv[1]).resolve(); raise SystemExit(0 if sys.version_info[:3] == (3, 12, 13) and sys.implementation.name == "cpython" and Path(sys.base_prefix).resolve().is_relative_to(root) else 1)'
"$managed_python" -c "$python_check" "$python_root" || fail "FD-PYTHON-VERIFY" "Managed CPython is not exactly CPython 3.12.13 below .tools/python."

venv_directory="$repository_root/.venv"
venv_python="$venv_directory/bin/python"
if [ -e "$venv_directory" ] && ! "$venv_python" -c "$venv_check" "$python_root" >/dev/null 2>&1; then
    [ "$venv_directory" = "$repository_root/.venv" ] || fail "FD-VENV-SAFETY" "Refusing to replace a virtual environment outside the repository root."
    rm -rf "$venv_directory" || fail "FD-VENV-REPLACE" "Incompatible repository-local .venv could not be removed."
fi

exec "$managed_python" "$bootstrap_path"
