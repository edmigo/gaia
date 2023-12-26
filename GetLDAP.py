import ldap3
from ldap3 import Server, Connection, SAFE_SYNC, SUBTREE

def GetLDAP3(userID):
    ldap3_server = 'ldap\\:elta.iai'
    ldap3_username = 'a99882@elta.iai'
    ldap3_password = 'vfr1234%^'

    username = userID

    server = Server(ldap3_server, port=389)
    conn = Connection(server, user=ldap3_username, password=ldap3_password, client_strategy=SAFE_SYNC)
    conn.bind()

    search_filter = '(uid={})'.format(username)
    search_base = 'cn=Users,dc=elta,dc=iai'
    search_scope = SUBTREE

    status, result, response, _ = conn.search(search_base=search_base, search_filter=search_filter, search_scope=search_scope, attributes='membersOf')

    myus = ''
    user_groups = []

    if status == True:
        pass

    return myus, user_groups

username, user_groups = GetLDAP3('a99882')
print('username: ' + username)
print('Qnt. Groups: ' + str(len(user_groups)))
