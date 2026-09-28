# VN97 Android debug build

This archive is produced only after Android compilation and the host contracts pass.
It is a developer test build signed with the CI debug key, not a production release.

## Contents

- `app-debug.apk`: Android test application.
- `build.json`: exact checked-out commit, workflow run and verification scope.
  For pull requests, the commit is the tested merge commit.
- `SHA256SUMS`: checksums for the APK, metadata and this document.

After extraction, run `sha256sum --check SHA256SUMS` in this directory.
Checksums detect download corruption; they do not establish publisher identity.
Download only from the intended VN97 GitHub Actions run.

## Installation and limits

Install on a test phone using Android's APK installer or
`adb install -r app-debug.apk`. CI debug signing keys can differ between runs.
An update may fail if an installed build has a different signing key.
Do not uninstall an existing app without preserving data you need:
uninstalling removes local app data and its device-backed credentials.

Model packages and broker credentials are not supplied by this archive.
A successful build does not establish model availability, inference quality,
phone performance, broker authentication, live trading authorization, or revenue.
The application may need its separately provisioned model before inference works.
Never put broker secrets in GitHub, issue comments, or chat.

Artifacts expire after seven days. These packages are for manual device validation;
this workflow does not publish an app-store release or execute trades.
