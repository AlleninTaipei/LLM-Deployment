# Unix-like Shell Commands

| **File and Directory Operations**   | **Description**                           | **Example**       |
|---------------|-------------------------------------------|-----------------------------------------|
| `cp`          | Copy files and directories                | `cp source.txt destination.txt`         |
| `mv`          | Move or rename files and directories      | `mv oldname.txt newname.txt`            |
| `rm`          | Remove files or directories               | `rm file.txt`                           |
| `mkdir`       | Create directories                        | `mkdir new_directory`                   |
| `rmdir`       | Remove empty directories                  | `rmdir empty_directory`                 |
| `ls`          | List directory contents                   | `ls -l`                                 |
| `chmod`       | Change file permissions                   | `chmod 755 script.sh`                   |
| `chown`       | Change file ownership                     | `chown user:group file.txt`             |
| `touch`       | Update file timestamps or create empty files | `touch newfile.txt`                  |

| **Text Processing**   | **Description**                           | **Example**                     |
|---------------|-------------------------------------------|-----------------------------------------|
| `cat`         | Concatenate and display file contents     | `cat file.txt`                          |
| `grep`        | Search text using patterns                | `grep 'pattern' file.txt`               |
| `sed`         | Stream editor for filtering and transforming text | `sed 's/old/new/g' file.txt`    |
| `awk`         | Pattern scanning and processing language  | `awk '{print $1}' file.txt`             |
| `cut`         | Remove sections from each line of files   | `cut -d':' -f1 /etc/passwd`             |
| `sort`        | Sort lines of text files                  | `sort file.txt`                         |
| `uniq`        | Report or omit repeated lines             | `uniq file.txt`                         |
| `tr`          | Translate or delete characters            | `tr 'a-z' 'A-Z' < file.txt`             |
| `head`        | Output the first part of files            | `head -n 10 file.txt`                   |
| `tail`        | Output the last part of files             | `tail -n 10 file.txt`                   |

| **System Information**   | **Description**                           | **Example**                  |
|---------------|-------------------------------------------|-----------------------------------------|
| `uname`       | Print system information                  | `uname -a`                              |
| `uptime`      | Tell how long the system has been running | `uptime`                                |
| `df`          | Report file system disk space usage       | `df -h`                                 |
| `du`          | Estimate file space usage                 | `du -sh *`                              |
| `free`        | Display amount of free and used memory    | `free -h`                               |
| `ps`          | Report a snapshot of current processes    | `ps aux`                                |
| `top`         | Display tasks and system resource usage   | `top`                                   |
| `id`          | Display user and group IDs                | `id username`                           |

| **Networking**   | **Description**                           | **Example**                          |
|---------------|-------------------------------------------|-----------------------------------------|
| `ping`        | Send ICMP ECHO_REQUEST to network hosts   | `ping example.com`                      |
| `wget`        | Non-interactive network downloader        | `wget https://example.com/file.txt`     |
| `curl`        | Transfer data from or to a server         | `curl https://example.com`              |
| `nc`          | Netcat, a versatile networking tool       | `nc -zv example.com 80`                 |
| `ftp`         | File Transfer Protocol client             | `ftp ftp.example.com`                   |
| `ssh`         | OpenSSH SSH client (remote login program) | `ssh user@example.com`                  |

| **File Compression and Archiving**   | **Description**                           | **Example**      |
|---------------|-------------------------------------------|-----------------------------------------|
| `tar`         | Archive files                             | `tar -cvf archive.tar directory/`       |
| `gzip`        | Compress files                            | `gzip file.txt`                         |
| `gunzip`      | Decompress files                          | `gunzip file.txt.gz`                    |
| `zip`         | Package and compress files                | `zip archive.zip file1 file2`           |
| `unzip`       | Extract compressed files from a ZIP archive | `unzip archive.zip`                   |

| **Shell Built-in Commands**   | **Description**                           | **Example**             |
|---------------|-------------------------------------------|-----------------------------------------|
| `echo`        | Display a line of text                    | `echo "Hello, World!"`                  |
| `cd`          | Change the shell working directory        | `cd /path/to/directory`                 |
| `exit`        | Exit the shell                            | `exit`                                  |
| `set`         | Set or unset shell options and positional parameters | `set -o noclobber`           |
| `unset`       | Unset values and attributes of shell variables | `unset VAR`                        |
| `export`      | Set an environment variable               | `export PATH=$PATH:/new/path`           |
| `alias`       | Create an alias for a command             | `alias ll='ls -la'`                     |
| `unalias`     | Remove an alias                           | `unalias ll`                            |
| `help`        | Display help for built-in commands        | `help cd`                               |

| **Process Management**   | **Description**                           | **Example**                  |
|---------------|-------------------------------------------|-----------------------------------------|
| `kill`        | Send a signal to a process                | `kill 1234`                             |
| `killall`     | Kill processes by name                    | `killall processname`                   |
| `jobs`        | List active jobs                          | `jobs`                                  |
| `fg`          | Bring a job to the foreground             | `fg %1`                                 |
| `bg`          | Place a job in the background             | `bg %1`                                 |
| `wait`        | Wait for a process to change state        | `wait 1234`                             |
| `ps`          | Report a snapshot of current processes    | `ps aux`                                |
| `top`         | Display tasks and system resource usage   | `top`                                   |


