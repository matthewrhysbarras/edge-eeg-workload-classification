# Checksums

SHA-256 checksums for release files can be regenerated with:

```powershell
Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch '\\.git\\' } | Sort-Object FullName | ForEach-Object {
  $hash = Get-FileHash -Algorithm SHA256 $_.FullName
  "$($hash.Hash.ToLower())  $($_.FullName.Replace((Get-Location).Path + '\\',''))"
}
```

The committed `checksums_sha256.txt` file records the checksums for the release artefacts.

