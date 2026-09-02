Option Explicit

Dim shell, root, scriptPath, command, videoPath
Set shell = CreateObject("WScript.Shell")
root = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
scriptPath = root & "\scripts\launch-system.ps1"
command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Hidden -File " & Chr(34) & scriptPath & Chr(34) & " -Action start -Mode simulated"
videoPath = shell.Environment("PROCESS")("DEEP_SEA_SIMULATION_VIDEO")
If videoPath <> "" Then
    command = command & " -VideoPath " & Chr(34) & videoPath & Chr(34)
End If
shell.Run command, 0, False
