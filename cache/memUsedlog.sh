#!/bin/bash                                                                                                                                      
MEM_LOGFILE="memory_used_log.csv"                                                                                                              
SERVER_LOGFILE="server_log.txt"

# Add headers to the memory usage CSV file                                                                                                       
echo "Timestamp,Used_Memory_GB" > "$MEM_LOGFILE"                                                                                            

# Loop to log memory usage and upload logs to different folders every 30 seconds                                                                         
while true; do                                                                                                                                   
    ### Memory Logging Section ###                                                                                                               
    # Capture the timestamp (YYYY-MM-DD HH:MM:SS)                                                                                                
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")                                                                                                       

    # Capture total and available memory in kB from /proc/meminfo                                                                                          
    TOTAL_MEM_KB=$(grep "MemTotal" /proc/meminfo | awk '{print $2}')                                                                     
    AVAILABLE_MEM_KB=$(grep "MemAvailable" /proc/meminfo | awk '{print $2}')                                                                     

    # Calculate used memory in kB and convert to GB                                                                                     
    USED_MEM_KB=$((TOTAL_MEM_KB - AVAILABLE_MEM_KB))
    USED_MEM_GB=$(echo "scale=2; $USED_MEM_KB / 1048576" | bc)

    # Log the used memory to the CSV file
    echo "$TIMESTAMP,$USED_MEM_GB" >> "$MEM_LOGFILE"                                                                                        
                                                                                                                                                 
    ### Google Drive Sync Section ###                                                                                                            
    # Sync memory log file to Google Drive folder for memory logs                                                                                
    #rclone copy "$MEM_LOGFILE" gdrive:CloudLab/Memory_Used --update                                                                            
    # Sync server log file to Google Drive folder for server logs                                                                                
    #rclone copy "$SERVER_LOGFILE" gdrive:CloudLab/Logs_Used --update                                                                           
                                                                                                                                                 
    # Wait for 30 seconds before the next iteration                                                                                              
    sleep 30                                                                                                                                     
done
