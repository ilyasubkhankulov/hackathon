# ssh -p 28831 root@209.137.198.14 -L 8080:localhost:8080

IP=209.137.198.14
PORT=28831

deploy:
	scp -P $(PORT) -r ./src/* root@$(IP):/home/hackathon/backend/
