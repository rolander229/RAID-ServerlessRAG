# PostgreSQL 只监听 127.0.0.1（本机回环），没有监听 0.0.0.0，所以外部 Pod 无法连接。
sed -i "s/#listen_addresses = 'localhost'/listen_addresses = '*'/" /etc/postgresql/14/main/postgresql.conf
grep listen_addresses /etc/postgresql/14/main/postgresql.conf
# 第二步：修改 pg_hba.conf 允许远程连接
echo "host  all  all  0.0.0.0/0  md5" >> /etc/postgresql/14/main/pg_hba.conf

# 方式一：pg_ctlcluster
pg_ctlcluster 14 main restart

# 方式二：service
service postgresql restart

netstat -tlnp | grep 5432
# tcp  0  0 0.0.0.0:5432  0.0.0.0:*  LISTEN  ✅