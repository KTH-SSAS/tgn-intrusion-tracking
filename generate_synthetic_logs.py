import argparse
import random
import pandas as pd
from datetime import datetime, timedelta

# Define actions and their applicable resource types
ACTIONS = {
    'list_all_buckets_of_project': 'project',
    'list_all_instances_in_project': 'project',
    'list_all_objects_in_bucket': 'bucket',
    'read_object': 'data_object',
    'log_on_to_instance': 'instance',
    'assume_service_account': 'service_account',
    'assign_ssh_key_to_instance': 'instance'
}

def generate_principals(num_users, num_service_accounts):
    users = [f'user_{i}' for i in range(1, num_users + 1)]
    service_accounts = [f'service_account_{i}' for i in range(1, num_service_accounts + 1)]
    return users, service_accounts

def generate_resources(num_projects, num_buckets, num_data_objects, num_instances):
    projects = [f'project_{i}' for i in range(1, num_projects + 1)]
    buckets = [f'bucket_{i}' for i in range(1, num_buckets + 1)]
    data_objects = [f'data_object_{i}' for i in range(1, num_data_objects + 1)]
    instances = [f'instance_{i}' for i in range(1, num_instances + 1)]
    return projects, buckets, data_objects, instances

def generate_random_timestamp(start_date, end_date):
    delta = end_date - start_date
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start_date + timedelta(seconds=random_seconds)

def generate_log_entries(num_entries, users, service_accounts, projects, buckets, data_objects, instances):
    principals = users + service_accounts
    actions = list(ACTIONS.keys())
    logs = []
    
    start_date = datetime.now() - timedelta(days=365)  # One year ago
    end_date = datetime.now()
    
    for _ in range(num_entries):
        principal = random.choice(principals)
        action = random.choice(actions)
        resource_type = ACTIONS[action]
        
        if resource_type == 'project':
            resource = random.choice(projects)
        elif resource_type == 'bucket':
            resource = random.choice(buckets)
        elif resource_type == 'data_object':
            resource = random.choice(data_objects)
        elif resource_type == 'instance':
            resource = random.choice(instances)
        elif resource_type == 'service_account':
            resource = random.choice(service_accounts)
        else:
            resource = 'unknown_resource'
        
        timestamp = generate_random_timestamp(start_date, end_date).isoformat() + 'Z'
        logs.append({
            'timestamp': timestamp,
            'principal': principal,
            'action': action,
            'resource': resource
        })
    
    return logs

def main():
    parser = argparse.ArgumentParser(description='Generate Synthetic GCP Cloud Audit Logs')
    parser.add_argument('--num_users', type=int, default=10, help='Number of user principals')
    parser.add_argument('--num_service_accounts', type=int, default=5, help='Number of service account principals')
    parser.add_argument('--num_projects', type=int, default=5, help='Number of projects')
    parser.add_argument('--num_buckets', type=int, default=20, help='Number of buckets')
    parser.add_argument('--num_data_objects', type=int, default=100, help='Number of data objects')
    parser.add_argument('--num_instances', type=int, default=50, help='Number of instances')
    parser.add_argument('--num_log_entries', type=int, default=10000, help='Number of log entries to generate')
    parser.add_argument('--output', type=str, default='synthetic_logs.csv', help='Output CSV file name')
    
    args = parser.parse_args()
    
    # Generate principals and resources
    users, service_accounts = generate_principals(args.num_users, args.num_service_accounts)
    projects, buckets, data_objects, instances = generate_resources(
        args.num_projects,
        args.num_buckets,
        args.num_data_objects,
        args.num_instances
    )
    
    # Generate log entries
    logs = generate_log_entries(
        args.num_log_entries,
        users,
        service_accounts,
        projects,
        buckets,
        data_objects,
        instances
    )
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(logs)
    df.to_csv(args.output, index=False)
    print(f'Successfully generated {args.num_log_entries} log entries and saved to {args.output}')

if __name__ == '__main__':
    main()
