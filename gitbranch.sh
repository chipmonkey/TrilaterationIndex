#! /bin/bash

# Thanks https://github.com/mafrosis/scripts/blob/master/git-create-branch.sh

# git-create-branch <branch_name>

function usage {
	echo 1>&2 "Usage: $0 branch_name"
	exit 127
}
 
if [ $# -ne 1 ]
then
	usage
fi
 
branch_name=$1
git fetch
echo "git checkout -b $branch_name"
git checkout -b $branch_name
echo "git push -u origin $branch_name"
git push -u origin $branch_name
git pull
